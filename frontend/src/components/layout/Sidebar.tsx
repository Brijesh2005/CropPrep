import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { useUiStore } from '@/store/uiStore';
import { cn } from '@/utils/cn';

const linkClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
    isActive
      ? 'bg-agriculture-600 text-white'
      : 'text-gray-700 hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-800'
  );

export function Sidebar() {
  const { isAuthenticated, isAdmin } = useAuth();
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const setSidebarOpen = useUiStore((s) => s.setSidebarOpen);

  const close = () => setSidebarOpen(false);

  const links = [
    { to: '/predict', label: 'Prediction', icon: '🎯' },
    { to: '/map', label: 'GIS Map', icon: '🗺️' },
    ...(isAuthenticated
      ? [
          { to: '/history', label: 'History', icon: '📜' },
          { to: '/dashboard', label: 'Dashboard', icon: '📊' },
          { to: '/profile', label: 'Profile', icon: '👤' },
          { to: '/settings', label: 'Settings', icon: '⚙️' },
        ]
      : []),
    ...(isAdmin ? [{ to: '/admin', label: 'Admin', icon: '🛡️' }] : []),
  ];

  return (
    <>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 bg-black/40 lg:hidden" onClick={close} aria-hidden />
      )}

      <aside
        className={cn(
          'fixed top-16 bottom-0 left-0 z-40 w-64 border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 transition-transform lg:static lg:translate-x-0 lg:z-0',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <nav className="flex flex-col gap-1">
          {links.map((link) => (
            <NavLink key={link.to} to={link.to} className={linkClass} onClick={close}>
              <span aria-hidden>{link.icon}</span>
              {link.label}
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  );
}

export default Sidebar;
