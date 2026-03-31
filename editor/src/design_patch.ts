import type { DownloadArtifact } from './state_io.js'
import type { AiCommand, DesignPatch, PatchOperation } from './slide_state.js'

export interface UpdatePatchInput {
  slideIndex: number
  elementId: string
  property: string
  value: unknown
  oldValue: unknown
  source?: 'human' | 'ai'
  timestamp?: number
}

export function createUpdatePatch(input: UpdatePatchInput): PatchOperation {
  return {
    op: 'update',
    path: `/slides/${input.slideIndex}/elements/${input.elementId}/${input.property}`,
    value: clonePatchValue(input.value),
    oldValue: clonePatchValue(input.oldValue),
    source: input.source ?? 'human',
    timestamp: input.timestamp ?? Date.now(),
  }
}

export function hasPatchValueChanged(oldValue: unknown, newValue: unknown): boolean {
  return serializePatchValue(oldValue) !== serializePatchValue(newValue)
}

export function createDesignPatch(
  operations: PatchOperation[],
  source: 'human' | 'ai' = 'human',
  timestamp = new Date().toISOString(),
): DesignPatch {
  return {
    timestamp,
    source,
    operations: operations.map(operation => clonePatchValue(operation)) as PatchOperation[],
  }
}

export function createDesignPatchDownload(operations: PatchOperation[]): DownloadArtifact {
  return {
    fileName: 'design_patch.json',
    mimeType: 'application/json;charset=utf-8',
    content: `${JSON.stringify(createDesignPatch(operations), null, 2)}\n`,
  }
}

function serializePatchValue(value: unknown): string {
  return JSON.stringify(clonePatchValue(value))
}

function clonePatchValue<T>(value: T): T {
  if (value === undefined) return value
  return JSON.parse(JSON.stringify(value)) as T
}

export function createAiCommandPatch(
  aiCommand: AiCommand,
  operations: PatchOperation[] = [],
): DesignPatch {
  return {
    timestamp: new Date().toISOString(),
    source: 'human',
    operations,
    aiCommand,
  }
}

export function createAiCommandDownload(
  aiCommand: AiCommand,
  operations: PatchOperation[] = [],
): DownloadArtifact {
  return {
    fileName: 'design_patch.json',
    mimeType: 'application/json;charset=utf-8',
    content: `${JSON.stringify(createAiCommandPatch(aiCommand, operations), null, 2)}
`,
  }
}
