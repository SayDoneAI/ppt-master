// index.ts — ppt-master editor 公共 API

export type {
  SlideState, Canvas, Slide, Element,
  TextElement, RectElement, PathElement, LineElement,
  CircleElement, ImageElement, GroupElement,
  Def, LinearGradientDef, FilterDef, GradientStop,
  DesignPatch, PatchOperation, AiCommand, AiCommandScope,
} from './slide_state.js'

export {
  slideToSvg, stateToSvgs,
  initPretext, layoutText,
  type TextLayoutResult,
} from './state_to_svg.js'

export {
  svgToSlide, svgsToState,
} from './svg_to_state.js'

export {
  applyDesignPatch,
  createAiCommand,
  createAiCommandDownload,
  createAiCommandPromptDownload,
  createAiCommandPatch,
  createDesignPatch,
  createDesignPatchDownload,
  createUpdatePatch,
  ensureDesignPatch,
  getDesignPatchPrimarySlideIndex,
  hasPatchValueChanged,
  parseDesignPatchJson,
} from './design_patch.js'

export {
  buildProjectStateFromSvgInputs,
  compareProjectFileName,
  createProjectSvgArtifacts,
  type ProjectSvgArtifact,
  type ProjectSvgInput,
} from './project_pipeline.js'

export {
  createHistoryEntry,
  flattenHistoryOperations,
  invertPatchOperation,
  type HistoryEntry,
} from './history.js'

export {
  createAppendSlidesPatch,
  createUniqueImportedSlides,
  ensureCompatibleCanvas,
} from './slide_import.js'
