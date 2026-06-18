import * as ort from 'onnxruntime-web';

export interface SonicSessionSummary {
  encoderInputs: string[];
  encoderOutputs: string[];
  decoderInputs: string[];
  decoderOutputs: string[];
}

export class SonicSessions {
  private encoder?: ort.InferenceSession;
  private decoder?: ort.InferenceSession;
  summary: SonicSessionSummary | null = null;

  async load(files: FileList | File[]) {
    const allFiles = Array.from(files);
    const encoderFile = allFiles.find((file) => /encoder/i.test(file.name));
    const decoderFile = allFiles.find((file) => /decoder/i.test(file.name));
    if (!encoderFile || !decoderFile) {
      throw new Error('Select both Sonic encoder and decoder ONNX files.');
    }

    const [encoderBytes, decoderBytes] = await Promise.all([
      encoderFile.arrayBuffer(),
      decoderFile.arrayBuffer(),
    ]);

    this.encoder = await ort.InferenceSession.create(encoderBytes, { executionProviders: ['wasm'] });
    this.decoder = await ort.InferenceSession.create(decoderBytes, { executionProviders: ['wasm'] });
    this.summary = {
      encoderInputs: [...this.encoder.inputNames],
      encoderOutputs: [...this.encoder.outputNames],
      decoderInputs: [...this.decoder.inputNames],
      decoderOutputs: [...this.decoder.outputNames],
    };
    return this.summary;
  }
}
