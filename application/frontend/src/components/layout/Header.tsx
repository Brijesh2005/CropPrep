import React from 'react';
import { Link, NavLink } from 'react-router-dom';
import { APP_NAME } from '@/config';
import { useAuth } from '@/hooks/useAuth';
import { useTheme } from '@/contexts/ThemeContext';
import { useUiStore } from '@/store/uiStore';
import { Button } from '@/components/ui/Button';
import { cn } from '@/utils/cn';

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'px-3 py-2 rounded-lg text-sm font-medium transition-colors',
    isActive
      ? 'bg-agriculture-100 text-agriculture-800 dark:bg-agriculture-900/40 dark:text-agriculture-300'
      : 'text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
  );

export function Header() {
  const { user, isAuthenticated, isAdmin, logout } = useAuth();
  const { theme, setTheme, resolved } = useTheme();
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);

  const cycleTheme = () => {
    const order = ['light', 'dark', 'system'] as const;
    const idx = order.indexOf(theme);
    setTheme(order[(idx + 1) % order.length]);
  };

  return (
    <header className="sticky top-0 z-40 border-b border-gray-200 dark:border-gray-800 bg-white/90 dark:bg-gray-900/90 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-4 px-4 sm:px-6">
        <button
          type="button"
          className="lg:hidden btn-ghost p-2"
          onClick={toggleSidebar}
          aria-label="Toggle navigation"
        >
          ☰
        </button>

        <Link to="/" className="flex items-center gap-2 shrink-0">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-agriculture-600 text-white text-lg font-bold">
            🌾
          </span>
          <span className="hidden sm:block text-lg font-bold text-gray-900 dark:text-white">
            {APP_NAME}
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-1 ml-4">
          <NavLink to="/predict" className={navLinkClass}>
            Predict
          </NavLink>
          <NavLink to="/map" className={navLinkClass}>
            Map
          </NavLink>
          {isAuthenticated && (
            <>
              <NavLink to="/history" className={navLinkClass}>
                History
              </NavLink>
              <NavLink to="/dashboard" className={navLinkClass}>
                Dashboard
              </NavLink>
            </>
          )}
          {isAdmin && (
            <NavLink to="/admin" className={navLinkClass}>
              Admin
            </NavLink>
          )}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={cycleTheme}
            aria-label={`Theme: ${theme}`}
            title={`Theme: ${theme} (${resolved})`}
          >
            {resolved === 'dark' ? '🌙' : '☀️'}
          </Button>

          {isAuthenticated ? (
            <>
              <Link
                to="/profile"
                className="hidden sm:flex items-center gap-2 text-sm text-gray-700 dark:text-gray-200 hover:text-agriculture-600"
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-full bg-agriculture-100 text-agriculture-700 dark:bg-agriculture-900 dark:text-agriculture-300 font-semibold">
                  {(user?.full_name || user?.email || '?')[0].toUpperCase()}
                </span>
                <span className="max-w-[120px] truncate">{user?.full_name || user?.email}</span>
              </Link>
              <Button variant="outline" size="sm" onClick={() => logout()}>
                Logout
              </Button>
            </>
          ) : (
            <>
              <Link to="/login">
                <Button variant="ghost" size="sm">
                  Login
                </Button>
              </Link>
              <Link to="/register">
                <Button variant="primary" size="sm">
                  Sign up
                </Button>
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;
