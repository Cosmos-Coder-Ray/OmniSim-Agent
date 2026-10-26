import type { ReactNode } from "react";

interface PlaceholderTabProps {
  title: string;
  children: ReactNode;
}

export function PlaceholderTab({ title, children }: PlaceholderTabProps) {
  return (
    <section className="placeholder-panel" aria-labelledby="placeholder-title">
      <div className="section-kicker">WORKSPACE</div>
      <h2 id="placeholder-title">{title}</h2>
      <p>{children}</p>
      <span className="placeholder-note">This view is not connected in this build.</span>
    </section>
  );
}
