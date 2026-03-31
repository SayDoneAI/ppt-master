// index.ts — ppt-master editor 公共 API

export type {
  SlideState, Canvas, Slide, Element,
  TextElement, RectElement, PathElement, LineElement,
  CircleElement, ImageElement, GroupElement,
  Def, LinearGradientDef, FilterDef, GradientStop,
  DesignPatch, PatchOperation,
} from './slide_state.js'

export {
  slideToSvg, stateToSvgs,
  initPretext, layoutText,
  type TextLayoutResult,
} from './state_to_svg.js'

export {
  svgToSlide, svgsToState,
} from './svg_to_state.js'
