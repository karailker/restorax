import {
  Sparkles,
  Film,
  Clapperboard,
  Newspaper,
  type LucideIcon,
} from "lucide-react";

export interface Preset {
  pipelineId: string;
  title: string;
  description: string;
  icon: LucideIcon;
  /** Tailwind classes for the icon accent. */
  accent: string;
}

export const PRESETS: Preset[] = [
  {
    pipelineId: "sr_x4",
    title: "4× Super-Resolution",
    description: "Upscale to 4× with detail recovery",
    icon: Sparkles,
    accent: "text-[var(--color-primary)]",
  },
  {
    pipelineId: "classic_film",
    title: "Classic Film",
    description: "Denoise & restore aged cinema reels",
    icon: Film,
    accent: "text-[var(--color-success)]",
  },
  {
    pipelineId: "vhs_restoration",
    title: "VHS Restoration",
    description: "Clean tracking lines & color bleed",
    icon: Clapperboard,
    accent: "text-[var(--color-warning)]",
  },
  {
    pipelineId: "newsreel",
    title: "Newsreel",
    description: "Stabilize & sharpen archival footage",
    icon: Newspaper,
    accent: "text-[var(--color-destructive)]",
  },
];
