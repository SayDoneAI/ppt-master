import { createDesignPatch } from './design_patch.js'
import type { DesignPatch, PatchOperation } from './slide_state.js'

export interface HistoryEntry {
  label: string
  forwardPatch: DesignPatch
  inversePatch: DesignPatch
}

export function createHistoryEntry(
  operations: PatchOperation[],
  source: 'human' | 'ai',
  label: string,
): HistoryEntry {
  return {
    label,
    forwardPatch: createDesignPatch(operations, source),
    inversePatch: createDesignPatch(
      [...operations].reverse().map(invertPatchOperation),
      source,
    ),
  }
}

export function flattenHistoryOperations(entries: readonly HistoryEntry[]): PatchOperation[] {
  return entries.flatMap(entry => entry.forwardPatch.operations.map(clonePatchOperation))
}

export function invertPatchOperation(operation: PatchOperation): PatchOperation {
  switch (operation.op) {
    case 'update':
      return {
        op: 'update',
        path: operation.path,
        source: operation.source,
        timestamp: operation.timestamp,
        value: cloneValue(operation.oldValue),
        oldValue: cloneValue(operation.value),
      }
    case 'add':
      return {
        op: 'delete',
        path: operation.path,
        oldValue: cloneValue(operation.value),
        source: operation.source,
        timestamp: operation.timestamp,
      }
    case 'delete':
      return {
        op: 'add',
        path: operation.path,
        value: cloneValue(operation.oldValue),
        source: operation.source,
        timestamp: operation.timestamp,
      }
    case 'reorder':
      return {
        op: 'reorder',
        path: operation.path,
        source: operation.source,
        timestamp: operation.timestamp,
        value: operation.oldValue,
        oldValue: operation.value,
      }
  }
}

function clonePatchOperation(operation: PatchOperation): PatchOperation {
  return cloneValue(operation)
}

function cloneValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}
