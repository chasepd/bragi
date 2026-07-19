import { BookOpen, Feather } from "lucide-react";

export function BrandLockup({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand-lockup ${compact ? "brand-lockup-compact" : ""}`}>
      <span className="brand-sigil" aria-hidden="true">
        <Feather size={compact ? 18 : 22} strokeWidth={1.7} />
      </span>
      <span className="brand-copy">
        <strong>Bragi</strong>
        <span>Living chronicle</span>
      </span>
    </div>
  );
}

export function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <BookOpen size={18} strokeWidth={1.8} />
    </span>
  );
}
