import { NavLink, Outlet } from "react-router-dom";
import { LayoutDashboard, Workflow, Film } from "lucide-react";
import { cn } from "@/lib/utils";
import { GpuStatusWidget } from "@/components/GpuStatusWidget";

const nav = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/builder", label: "Pipeline Builder", icon: Workflow, end: false },
];

export function AppShell() {
  return (
    <div className="flex h-full">
      <aside className="flex w-60 shrink-0 flex-col border-r border-[var(--color-border)] bg-[var(--color-card)]">
        <div className="flex items-center gap-2 px-5 py-4">
          <Film className="size-5 text-[var(--color-primary)]" />
          <span className="text-lg font-semibold tracking-tight">RestoraX</span>
        </div>
        <nav className="flex flex-1 flex-col gap-1 px-3">
          {nav.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-[var(--color-primary)]/15 text-[var(--color-primary)]"
                    : "text-[var(--color-muted-foreground)] hover:bg-[var(--color-accent)] hover:text-[var(--color-foreground)]",
                )
              }
            >
              <Icon className="size-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-3">
          <GpuStatusWidget />
        </div>
      </aside>
      <main className="h-full flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
