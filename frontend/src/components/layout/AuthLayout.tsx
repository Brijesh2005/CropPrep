import React from 'react';
import { Link } from 'react-router-dom';
import { APP_NAME } from '@/config';

export function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-agriculture-50 via-white to-agriculture-100 dark:from-gray-950 dark:via-gray-900 dark:to-agriculture-950">
      <div className="p-6">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-lg font-bold text-gray-900 dark:text-white"
        >
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-agriculture-600 text-white">
            🌾
          </span>
          {APP_NAME}
        </Link>
      </div>
      <div className="flex-1 flex items-center justify-center px-4 pb-12">
        <div className="w-full max-w-md">{children}</div>
      </div>
    </div>
  );
}

export default AuthLayout;
