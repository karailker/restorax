"use client";

import {
  ReactCompareSlider,
  ReactCompareSliderImage,
  ReactCompareSliderHandle,
} from "react-compare-slider";

interface Props {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
}

export default function CompareSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "Original",
  afterLabel = "Restored",
}: Props) {
  return (
    <div className="relative rounded-xl overflow-hidden shadow-lg">
      <ReactCompareSlider
        handle={
          <ReactCompareSliderHandle
            buttonStyle={{ backdropFilter: "blur(4px)", background: "rgba(255,255,255,0.8)" }}
          />
        }
        itemOne={
          <div className="relative">
            <ReactCompareSliderImage src={beforeSrc} alt={beforeLabel} />
            <span className="absolute top-3 left-3 bg-black/60 text-white text-xs px-2 py-1 rounded">
              {beforeLabel}
            </span>
          </div>
        }
        itemTwo={
          <div className="relative">
            <ReactCompareSliderImage src={afterSrc} alt={afterLabel} />
            <span className="absolute top-3 right-3 bg-indigo-600/80 text-white text-xs px-2 py-1 rounded">
              {afterLabel}
            </span>
          </div>
        }
        style={{ width: "100%", aspectRatio: "16/9" }}
      />
    </div>
  );
}
