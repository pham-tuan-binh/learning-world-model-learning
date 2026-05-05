import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BookOpen, GitFork } from "lucide-react";
import {
  ASSET_DIR,
  buildActionTable,
  createPreviewDecoder,
  createPreviewDynamics,
  createPreviewManifest,
  createPreviewSeeds,
  loadJson,
  loadRealRuntime,
  makeFloatTensor,
  makeInt64Tensor,
  reloadWasmRuntime,
  shouldUsePreviewMode,
} from "./player.js";

const HOLD_REPEAT_DELAY_MS = 60;

function flattenFrames(frames) {
  return frames.flatMap((frame) => frame);
}

function flattenActions(actions) {
  return actions.flatMap((action) => action);
}

function drawFrame(canvas, manifest, frameTensor) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const width = manifest.frameSize;
  const height = manifest.frameSize;
  const image = ctx.createImageData(width, height);
  const data = frameTensor.data;
  const plane = width * height;

  for (let idx = 0; idx < plane; idx += 1) {
    const r = Math.max(0, Math.min(255, Math.round(data[idx] * 255)));
    const g = Math.max(0, Math.min(255, Math.round(data[plane + idx] * 255)));
    const b = Math.max(0, Math.min(255, Math.round(data[2 * plane + idx] * 255)));
    const p = idx * 4;
    image.data[p] = r;
    image.data[p + 1] = g;
    image.data[p + 2] = b;
    image.data[p + 3] = 255;
  }

  ctx.putImageData(image, 0, 0);
}

export default function App() {
  const canvasRef = useRef(null);
  const runtimeRef = useRef({ dynamics: null, decoder: null });
  const tokensRef = useRef([]);
  const actionsRef = useRef([]);
  const busyRef = useRef(false);
  const heldActionRef = useRef(null);
  const holdLoopRunningRef = useRef(false);
  const sessionRecoveringRef = useRef(false);

  const [manifest, setManifest] = useState(null);
  const [seeds, setSeeds] = useState([]);
  const [seedIndex, setSeedIndex] = useState(0);
  const [currentActionId, setCurrentActionId] = useState(0);
  const [steps, setSteps] = useState(0);
  const [status, setStatus] = useState("Loading assets...");
  const [previewMode, setPreviewMode] = useState(false);
  const [loading, setLoading] = useState({
    active: true,
    label: "Loading assets...",
    percent: null,
  });

  const actions = useMemo(() => manifest?.actions ?? [], [manifest]);

  const decodeCurrentFrame = useCallback(async () => {
    if (!manifest || !canvasRef.current || tokensRef.current.length === 0) {
      return;
    }
    const current = tokensRef.current[tokensRef.current.length - 1];
    const input = makeInt64Tensor(current, [1, 1, manifest.numPatches], previewMode);
    const outputs = await runtimeRef.current.decoder.run({ indices: input });
    drawFrame(canvasRef.current, manifest, outputs.frames);
  }, [manifest, previewMode]);

  const resetToSeed = useCallback(
    async (nextSeedIndex = seedIndex) => {
      if (!seeds.length) {
        return;
      }
      const seed = seeds[nextSeedIndex] ?? seeds[0];
      tokensRef.current = [seed.tokens.slice()];
      actionsRef.current = [];
      setSteps(0);
      try {
        await decodeCurrentFrame();
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error));
      }
      setLoading((current) => (current.active ? { ...current, active: false } : current));
    },
    [decodeCurrentFrame, seedIndex, seeds],
  );

  const stepModel = useCallback(
    async (actionId) => {
      if (!manifest || busyRef.current || tokensRef.current.length === 0) {
        return;
      }

      busyRef.current = true;
      try {
        const startedAt = performance.now();
        const action = actions.find((item) => item.id === actionId) ?? actions[0];
        actionsRef.current.push(action.vector);

        const contextFrames = tokensRef.current.length;
        const tokenTensor = makeInt64Tensor(
          flattenFrames(tokensRef.current),
          [1, contextFrames, manifest.numPatches],
          previewMode,
        );
        const actionTensor = makeFloatTensor(
          flattenActions(actionsRef.current),
          [1, contextFrames, manifest.actionDim],
          previewMode,
        );

        const outputs = await runtimeRef.current.dynamics.run({
          tokens: tokenTensor,
          actions: actionTensor,
        });
        const next = Array.from(outputs.next_tokens.data, Number);
        tokensRef.current.push(next);

        if (tokensRef.current.length > manifest.maxContextFrames) {
          tokensRef.current.shift();
          actionsRef.current.shift();
        }

        setSteps((value) => value + 1);
        await decodeCurrentFrame();
        const elapsedMs = Math.round(performance.now() - startedAt);
        setStatus(`Ready (${runtimeRef.current.backend}, ${elapsedMs}ms)`);
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error));
        if (!previewMode && !sessionRecoveringRef.current && manifest) {
          sessionRecoveringRef.current = true;
          setStatus("Session error — reloading WASM...");
          try {
            runtimeRef.current = await reloadWasmRuntime(manifest);
            setStatus(`Recovered (${runtimeRef.current.backend})`);
          } catch (recoveryError) {
            setStatus("Recovery failed — please reload the page.");
          } finally {
            sessionRecoveringRef.current = false;
          }
        }
      } finally {
        busyRef.current = false;
      }
    },
    [actions, decodeCurrentFrame, manifest, previewMode],
  );

  const stopHoldingAction = useCallback(() => {
    heldActionRef.current = null;
  }, []);

  const startHoldingAction = useCallback(
    (actionId) => {
      if (!manifest || loading.active) {
        return;
      }

      heldActionRef.current = actionId;
      setCurrentActionId(actionId);

      if (holdLoopRunningRef.current) {
        return;
      }

      holdLoopRunningRef.current = true;
      const runHeldAction = async () => {
        while (heldActionRef.current !== null) {
          const heldActionId = heldActionRef.current;
          setCurrentActionId(heldActionId);
          await stepModel(heldActionId);
          if (heldActionRef.current === null) {
            break;
          }
          await new Promise((resolve) => {
            window.setTimeout(resolve, HOLD_REPEAT_DELAY_MS);
          });
        }
        holdLoopRunningRef.current = false;
      };

      void runHeldAction();
    },
    [loading.active, manifest, stepModel],
  );

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      const forcePreview = shouldUsePreviewMode();
      try {
        if (!cancelled) {
          setLoading({ active: true, label: "Loading assets...", percent: 2 });
        }
        if (forcePreview) {
          const previewManifest = createPreviewManifest();
          const previewSeeds = createPreviewSeeds(previewManifest);
          runtimeRef.current = {
            dynamics: createPreviewDynamics(previewManifest),
            decoder: createPreviewDecoder(previewManifest),
          };
          if (!cancelled) {
            setPreviewMode(true);
            setManifest(previewManifest);
            setSeeds(previewSeeds);
            setStatus("Preview mode - no model loaded");
            setLoading({ active: false, label: "", percent: null });
          }
          return;
        }

        if (!cancelled) {
          setLoading({ active: true, label: "Loading manifest", percent: 5 });
        }
        const loadedManifest = await loadJson(`${ASSET_DIR}/manifest.json`);
        loadedManifest.actions = buildActionTable(loadedManifest);
        if (!cancelled) {
          setLoading({ active: true, label: "Loading seeds", percent: 8 });
        }
        const seedPayload = await loadJson(`${ASSET_DIR}/seeds.json`);
        const loadedSeeds = seedPayload.seeds ?? [];
        if (loadedSeeds.length === 0) {
          throw new Error("No seed tokens found in assets/seeds.json");
        }
        const runtime = await loadRealRuntime(loadedManifest, (progress) => {
          if (cancelled) {
            return;
          }
          setLoading({
            active: true,
            label: progress.label,
            percent: progress.percent,
          });
        });

        if (!cancelled) {
          runtimeRef.current = runtime;
          setPreviewMode(false);
          setManifest(loadedManifest);
          setSeeds(loadedSeeds);
          setStatus(`Ready (${runtime.backend})`);
        }
      } catch (error) {
        const previewManifest = createPreviewManifest();
        const previewSeeds = createPreviewSeeds(previewManifest);
        runtimeRef.current = {
          dynamics: createPreviewDynamics(previewManifest),
          decoder: createPreviewDecoder(previewManifest),
        };
        if (!cancelled) {
          setPreviewMode(true);
          setManifest(previewManifest);
          setSeeds(previewSeeds);
          setStatus("Preview mode - no exported assets found");
          setLoading({ active: false, label: "", percent: null });
        }
      }
    }

    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (manifest && seeds.length > 0) {
      void resetToSeed(seedIndex);
    }
  }, [manifest, resetToSeed, seedIndex, seeds.length]);

  useEffect(() => {
    function onKeyDown(event) {
      const keyMap = {
        ArrowUp: 0,
        KeyW: 0,
        ArrowLeft: 1,
        KeyA: 1,
        ArrowRight: 2,
        KeyD: 2,
        ArrowDown: 3,
        KeyS: 3,
        Space: 3,
        Digit1: 0,
        Digit2: 1,
        Digit3: 2,
        Digit4: 3,
      };

      if (!(event.code in keyMap) || !manifest) {
        return;
      }
      event.preventDefault();
      const actionId = keyMap[event.code] % manifest.nActions;
      setCurrentActionId(actionId);
      void stepModel(actionId);
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [manifest, stepModel]);

  useEffect(() => {
    window.addEventListener("pointerup", stopHoldingAction);
    window.addEventListener("pointercancel", stopHoldingAction);
    window.addEventListener("blur", stopHoldingAction);
    return () => {
      window.removeEventListener("pointerup", stopHoldingAction);
      window.removeEventListener("pointercancel", stopHoldingAction);
      window.removeEventListener("blur", stopHoldingAction);
    };
  }, [stopHoldingAction]);

  const frameSize = manifest?.frameSize ?? 128;
  const loadingPercent =
    typeof loading.percent === "number"
      ? Math.max(0, Math.min(100, Math.round(loading.percent)))
      : null;

  return (
    <main className="flex min-h-screen w-full justify-center px-4 py-6 pb-16 sm:px-6 sm:py-10 lg:py-12">
      <section className="w-full max-w-[640px] select-none" aria-label="World model player">
        <header className="mb-6 text-center">
          <img
            src="./assets/doom-1993.gif"
            alt="Doom marine idle animation"
            className="mx-auto h-20 w-auto [image-rendering:pixelated] sm:h-24"
          />
          <h1 className="mt-3 text-xl font-normal lowercase leading-snug tracking-normal text-dark">
            Learning World Model Learning
          </h1>
          <div className="mt-3 flex justify-center gap-2">
            <a
              className="inline-flex h-8 items-center gap-1.5 rounded-sm border border-border bg-paper px-3 font-mono text-[11px] font-medium uppercase leading-none tracking-[0.5px] text-mid hover:border-cloudy hover:bg-hover hover:text-dark focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-dark"
              href="https://github.com/pham-tuan-binh/learning-world-model-learning"
            >
              <GitFork className="h-3.5 w-3.5" aria-hidden="true" strokeWidth={1.8} />
              GitHub
            </a>
            <a
              className="inline-flex h-8 items-center gap-1.5 rounded-sm border border-border bg-paper px-3 font-mono text-[11px] font-medium uppercase leading-none tracking-[0.5px] text-mid hover:border-cloudy hover:bg-hover hover:text-dark focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-dark"
              href="https://garden.binhph.am/articles/learning-world-model-learning"
            >
              <BookOpen className="h-3.5 w-3.5" aria-hidden="true" strokeWidth={1.8} />
              Blog
            </a>
          </div>
        </header>

        <div className="relative aspect-square w-full overflow-hidden rounded-sm border border-border bg-black">
          <canvas
            ref={canvasRef}
            id="screen"
            width={frameSize}
            height={frameSize}
            className="block h-full w-full bg-black [image-rendering:pixelated]"
          />

          <div className="absolute right-2.5 top-2.5 flex overflow-hidden rounded-sm border border-paper/15 bg-black/55 backdrop-blur-[2px]">
            <select
              aria-label="Seed"
              className="h-7 w-32 border-0 bg-black/35 px-2 font-mono text-[10px] font-medium uppercase tracking-[0.5px] text-paper/80 outline-none hover:bg-black/50 focus:bg-black/50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-paper"
              value={seedIndex}
              onChange={(event) => setSeedIndex(Number(event.target.value))}
            >
              {seeds.map((seed, index) => (
                <option value={index} key={seed.name ?? index}>
                  {seed.name ?? `Seed ${index + 1}`}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="h-7 w-20 border-0 border-l border-paper/15 bg-black/35 px-2.5 font-mono text-[10px] font-medium uppercase tracking-[0.5px] text-paper/80 hover:bg-black/50 focus:bg-black/50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-paper"
              onClick={() => void resetToSeed(seedIndex)}
            >
              Reset
            </button>
          </div>

          <div className="pointer-events-none absolute inset-x-4 bottom-3 flex justify-between gap-3 font-mono text-[10px] leading-[1.4] tracking-[0.5px] text-paper/75">
            <span>{status}</span>
            <span>{steps}</span>
          </div>

          {loading.active ? (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/75 px-8 text-paper backdrop-blur-[1px]">
              <div className="w-full max-w-[280px]">
                <div className="font-mono text-[10px] font-medium uppercase leading-[1.4] tracking-[0.8px] text-paper/80">
                  {loading.label}
                </div>
                <div className="mt-3 h-1.5 w-full overflow-hidden rounded-sm bg-paper/15">
                  <div
                    className={[
                      "h-full rounded-sm bg-paper/80 transition-[width] duration-200",
                      loadingPercent === null ? "w-1/3 animate-pulse" : "",
                    ].join(" ")}
                    style={
                      loadingPercent === null
                        ? undefined
                        : { width: `${loadingPercent}%` }
                    }
                  />
                </div>
                <div className="mt-2 flex justify-between font-mono text-[10px] leading-[1.4] tracking-[0.5px] text-paper/55">
                  <span>Model loading</span>
                  <span>{loadingPercent === null ? "..." : `${loadingPercent}%`}</span>
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div className="mt-3 grid grid-cols-4 gap-2">
          {actions.map((action) => {
            const isActive = currentActionId === action.id;
            return (
              <button
                key={action.id}
                type="button"
                className={[
                  "h-10 rounded-sm border font-mono text-[10px] font-medium uppercase tracking-[0.5px]",
                  "select-none [touch-action:manipulation] focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-dark",
                  isActive
                    ? "border-cloudy bg-pampas text-dark hover:border-cloudy hover:bg-pampas"
                    : "border-border bg-paper text-mid hover:border-cloudy hover:bg-hover focus:bg-paper",
                ].join(" ")}
                onClick={(event) => {
                  if (event.detail === 0) {
                    setCurrentActionId(action.id);
                    void stepModel(action.id);
                  }
                }}
                onPointerDown={(event) => {
                  event.preventDefault();
                  event.currentTarget.setPointerCapture(event.pointerId);
                  startHoldingAction(action.id);
                }}
                onPointerLeave={() => {
                  stopHoldingAction();
                }}
                onPointerUp={(event) => {
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId);
                  }
                  stopHoldingAction();
                }}
                onPointerCancel={() => {
                  stopHoldingAction();
                }}
              >
                {action.label ?? `A${action.id}`}
              </button>
            );
          })}
        </div>

        <article className="mt-6 select-text text-base leading-7 text-dark sm:text-lg sm:leading-8">
          <p>
            This is a 7M parameter video world model trained on 100 episodes of Doom
            gameplay generated with VizDoom. You choose an action, and the model
            predicts what the next frame should look like. It was trained on only
            videos of Doom, without action labels, which means the same approach could work on any large
            video collection, without needing labels.
          </p>

          <p className="mt-4">
            The model was trained as part of{" "}
            <a
              className="font-medium text-blue underline decoration-border underline-offset-4 hover:decoration-blue"
              href="https://garden.binhph.am/articles/learning-world-model-learning"
            >
              Learning World Model Learning
            </a>
            ,
            a world model learning series written by{" "}
            <a
              className="font-medium text-blue underline decoration-border underline-offset-4 hover:decoration-blue"
              href="https://github.com/pham-tuan-binh"
            >
              Binh Pham
            </a>
            . The series began in January 2026 and concluded with its final article in
            May 2026. The author wrote it to help himself and others understand what it
            takes to train a Genie-based video world model.
          </p>

          <p className="mt-4">The series is divided into four major articles:</p>

          <ul className="mt-3 list-disc space-y-2 pl-5">
            <li>
              <span className="font-medium text-dark">Introduction</span> covers the
              intuition and architecture of a simple world model.{" "}
              <a
                className="font-medium text-blue underline decoration-border underline-offset-4 hover:decoration-blue"
                href="https://github.com/pham-tuan-binh/learning-world-model-learning"
              >
                Read it here.
              </a>
            </li>
            <li>
              <span className="font-medium text-dark">Video Tokenizer</span> covers how
              world models tokenize videos for downstream processing.{" "}
              <a
                className="font-medium text-blue underline decoration-border underline-offset-4 hover:decoration-blue"
                href="https://github.com/pham-tuan-binh/learning-world-model-learning/tree/main/1.video-tokenizer"
              >
                Read it here.
              </a>
            </li>
            <li>
              <span className="font-medium text-dark">Inverse Dynamics</span> covers how
              world models can learn from raw videos without action annotations.{" "}
              <a
                className="font-medium text-blue underline decoration-border underline-offset-4 hover:decoration-blue"
                href="https://github.com/pham-tuan-binh/learning-world-model-learning/tree/main/2.inverse-dynamics"
              >
                Read it here.
              </a>
            </li>
            <li>
              <span className="font-medium text-dark">Dynamics</span> covers how world
              models learn to predict the future from current actions and states.{" "}
              <a
                className="font-medium text-blue underline decoration-border underline-offset-4 hover:decoration-blue"
                href="https://github.com/pham-tuan-binh/learning-world-model-learning/tree/main/3.dynamics"
              >
                Read it here.
              </a>
            </li>
          </ul>
        </article>
      </section>
    </main>
  );
}
