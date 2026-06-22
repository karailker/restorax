import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { LayoutDashboard, Workflow, Film, Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { GpuStatusWidget } from "@/components/GpuStatusWidget";

const nav = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/builder", label: "Pipeline Builder", icon: Workflow, end: false },
];

export function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex h-full flex-col md:flex-row">
      <header className="flex items-center justify-between border-b border-border bg-card px-4 py-3 md:hidden">
        <div className="flex items-center gap-2">
          <Film className="size-5 text-primary" />
          <span className="text-lg font-semibold tracking-tight">RestoraX</span>
        </div>
        <button
          type="button"
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation"
          className="rounded-md p-2 text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <Menu className="size-5" />
        </button>
      </header>

      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-60 shrink-0 flex-col border-r border-border bg-card transition-transform md:static md:z-auto md:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center justify-between gap-2 px-5 py-4">
          <div className="flex items-center gap-2">
            <Film className="size-5 text-primary" />
            <span className="text-lg font-semibold tracking-tight">RestoraX</span>
          </div>
          <button
            type="button"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
            className="rounded-md p-2 text-muted-foreground hover:bg-accent hover:text-foreground md:hidden"
          >
            <X className="size-5" />
          </button>
        </div>
        <nav className="flex flex-1 flex-col gap-1 px-3">
          {nav.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary/15 text-primary"
                    : "text-muted-foreground hover:bg-accent hover:text-foreground",
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
