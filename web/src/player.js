import * as ort from "onnxruntime-web/webgpu";

export const ASSET_DIR = "./assets";
export const ORT_VERSION = "1.25.1";

export function shouldUsePreviewMode() {
  const params = new URLSearchParams(window.location.search);
  return params.has("preview") || window.location.protocol === "file:";
}

export function actionVector(actionId, actionDim) {
  const values = new Array(actionDim);
  for (let bit = 0; bit < actionDim; bit += 1) {
    values[bit] = (actionId >> bit) & 1 ? 1 : -1;
  }
  return values;
}

export function buildActionTable(manifest) {
  if (Array.isArray(manifest.actions) && manifest.actions.length > 0) {
    return manifest.actions;
  }

  return Array.from({ length: manifest.nActions }, (_, id) => ({
    id,
    label: `A${id}`,
    vector: actionVector(id, manifest.actionDim),
  }));
}

export function createPreviewManifest() {
  const manifest = {
    formatVersion: 1,
    frameSize: 128,
    numPatches: 256,
    gridSize: 16,
    vocabSize: 1024,
    actionDim: 3,
    nActions: 4,
    maxContextFrames: 4,
  };
  return { ...manifest, actions: buildActionTable(manifest) };
}

export function createPreviewSeeds(manifest) {
  return Array.from({ length: 4 }, (_, seedIndex) => {
    const tokens = Array.from({ length: manifest.numPatches }, (_, idx) => {
      const x = idx % manifest.gridSize;
      const y = Math.floor(idx / manifest.gridSize);
      return (x * 41 + y * 67 + seedIndex * 173) % manifest.vocabSize;
    });
    return { name: `Preview ${seedIndex + 1}`, tokens };
  });
}

export function createPreviewDynamics(manifest) {
  return {
    async run(inputs) {
      const tokenData = Array.from(inputs.tokens.data, Number);
      const actionData = Array.from(inputs.actions.data, Number);
      const current = tokenData.slice(-manifest.numPatches);
      const latestAction = actionData.slice(-manifest.actionDim);
      const actionOffset = latestAction.reduce((sum, value, idx) => {
        return sum + (value > 0 ? idx + 1 : -(idx + 1));
      }, 0);
      const next = current.map((token, idx) => {
        const x = idx % manifest.gridSize;
        const y = Math.floor(idx / manifest.gridSize);
        return (token + actionOffset * 13 + x * 3 + y * 5 + manifest.vocabSize) % manifest.vocabSize;
      });
      return { next_tokens: { data: BigInt64Array.from(next, BigInt) } };
    },
  };
}

export function createPreviewDecoder(manifest) {
  return {
    async run(inputs) {
      const width = manifest.frameSize;
      const height = manifest.frameSize;
      const plane = width * height;
      const tokens = Array.from(inputs.indices.data, Number).slice(-manifest.numPatches);
      const data = new Float32Array(plane * 3);

      for (let py = 0; py < height; py += 1) {
        for (let px = 0; px < width; px += 1) {
          const tx = Math.floor(px / 8);
          const ty = Math.floor(py / 8);
          const token = tokens[ty * manifest.gridSize + tx] ?? 0;
          const noise = ((token % 37) / 36 - 0.5) * 0.08;
          const horizon = py < height * 0.45;
          const wallBand = Math.abs(px - width / 2) / (width / 2);
          const floorDepth = Math.max(0, (py - height * 0.45) / (height * 0.55));
          const muzzle = Math.max(0, 1 - Math.hypot(px - width * 0.53, py - height * 0.76) / 24);
          const barrel = px > width * 0.45 && px < width * 0.66 && py > height * 0.68;
          const vignette = Math.hypot(px - width / 2, py - height / 2) / (width * 0.7);
          const idx = py * width + px;
          let r = horizon ? 0.12 + noise : 0.19 + floorDepth * 0.16 + noise;
          let g = horizon ? 0.11 + noise * 0.6 : 0.13 + floorDepth * 0.08 + noise * 0.4;
          let b = horizon ? 0.1 + noise * 0.4 : 0.08 + floorDepth * 0.04;

          if (!horizon && wallBand > 0.78) {
            r = 0.12 + noise;
            g = 0.09 + noise * 0.5;
            b = 0.07;
          }

          if (barrel) {
            r = 0.06 + muzzle * 0.65;
            g = 0.055 + muzzle * 0.32;
            b = 0.05;
          }

          data[idx] = Math.max(0, Math.min(1, r - vignette * 0.12));
          data[plane + idx] = Math.max(0, Math.min(1, g - vignette * 0.1));
          data[2 * plane + idx] = Math.max(0, Math.min(1, b - vignette * 0.08));
        }
      }

      return { frames: { data } };
    },
  };
}

export function makeInt64Tensor(data, dims, previewMode) {
  if (previewMode) {
    return { data: BigInt64Array.from(data, BigInt), dims };
  }
  return new ort.Tensor("int64", BigInt64Array.from(data, BigInt), dims);
}

export function makeFloatTensor(data, dims, previewMode) {
  if (previewMode) {
    return { data: Float32Array.from(data), dims };
  }
  return new ort.Tensor("float32", Float32Array.from(data), dims);
}

function reportProgress(onProgress, label, percent = null) {
  if (!onProgress) {
    return;
  }
  onProgress({ label, percent });
}

async function fetchModelBytes(modelName, onChunk) {
  const response = await fetch(`${ASSET_DIR}/${modelName}`);
  if (!response.ok) {
    throw new Error(`Missing ${ASSET_DIR}/${modelName}`);
  }

  const total = Number(response.headers.get("content-length")) || null;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onChunk(buffer.byteLength, buffer.byteLength);
    return new Uint8Array(buffer);
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loaded += value.byteLength;
    onChunk(loaded, total);
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

export async function loadRealRuntime(manifest, onProgress) {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

  const canUseWebGpu =
    Boolean(navigator.gpu) && manifest.webgpuDynamicsModel && manifest.webgpuDecoderModel;

  async function createSessions(dynamicsModel, decoderModel, executionProviders, backend) {
    const loadedByModel = new Map();
    const totalByModel = new Map();

    function onChunk(modelName, loaded, total) {
      loadedByModel.set(modelName, loaded);
      if (total) {
        totalByModel.set(modelName, total);
      }

      const loadedBytes = Array.from(loadedByModel.values()).reduce((sum, value) => sum + value, 0);
      const totalBytes = Array.from(totalByModel.values()).reduce((sum, value) => sum + value, 0);
      const percent = totalBytes > 0 ? 10 + Math.min(70, (loadedBytes / totalBytes) * 70) : null;
      reportProgress(onProgress, `Downloading ${backend} models`, percent);
    }

    reportProgress(onProgress, `Downloading ${backend} models`, 10);
    const [dynamicsBytes, decoderBytes] = await Promise.all([
      fetchModelBytes(dynamicsModel, (loaded, total) => onChunk(dynamicsModel, loaded, total)),
      fetchModelBytes(decoderModel, (loaded, total) => onChunk(decoderModel, loaded, total)),
    ]);

    reportProgress(onProgress, `Initializing ${backend}`, 85);
    let dynamics;
    let decoder;
    if (backend === "WebGPU") {
      dynamics = await ort.InferenceSession.create(dynamicsBytes, {
        executionProviders,
      });
      reportProgress(onProgress, `Initializing ${backend}`, 92);
      decoder = await ort.InferenceSession.create(decoderBytes, {
        executionProviders,
      });
    } else {
      [dynamics, decoder] = await Promise.all([
        ort.InferenceSession.create(dynamicsBytes, {
          executionProviders,
        }),
        ort.InferenceSession.create(decoderBytes, {
          executionProviders,
        }),
      ]);
    }
    reportProgress(onProgress, `Ready (${backend})`, 100);
    return { dynamics, decoder };
  }

  if (canUseWebGpu) {
    try {
      return await createSessions(
        manifest.webgpuDynamicsModel,
        manifest.webgpuDecoderModel,
        ["webgpu", "wasm"],
        "WebGPU",
      ).then((runtime) => ({ ...runtime, backend: "WebGPU" }));
    } catch (error) {
      console.warn("WebGPU model loading failed; falling back to WASM.", error);
      reportProgress(onProgress, "WebGPU failed; loading WASM fallback", 5);
    }
  }

  return createSessions(manifest.dynamicsModel, manifest.decoderModel, ["wasm"], "WASM").then(
    (runtime) => ({
      ...runtime,
      backend: "WASM",
    }),
  );
}

export async function loadJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Missing ${url}`);
  }
  return response.json();
}
