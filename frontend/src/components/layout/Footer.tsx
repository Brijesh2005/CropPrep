import React from 'react';
import { Link } from 'react-router-dom';
import { APP_NAME, APP_VERSION } from '@/config';

export function Footer() {
  return (
    <footer className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 mt-auto">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 py-8">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
          <div>
            <p className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
              <span>🌾</span> {APP_NAME}
            </p>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              AI-powered multi-crop recommendation and yield prediction for Indian agriculture.
            </p>
          </div>
          <div>
            <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Product</p>
            <ul className="space-y-1 text-sm text-gray-500 dark:text-gray-400">
              <li>
                <Link to="/predict" className="hover:text-agriculture-600">
                  Predict
                </Link>
              </li>
              <li>
                <Link to="/map" className="hover:text-agriculture-600">
                  Map
                </Link>
              </li>
              <li>
                <Link to="/dashboard" className="hover:text-agriculture-600">
                  Dashboard
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Account</p>
            <ul className="space-y-1 text-sm text-gray-500 dark:text-gray-400">
              <li>
                <Link to="/profile" className="hover:text-agriculture-600">
                  Profile
                </Link>
              </li>
              <li>
                <Link to="/settings" className="hover:text-agriculture-600">
                  Settings
                </Link>
              </li>
              <li>
                <Link to="/history" className="hover:text-agriculture-600">
                  History
                </Link>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-8 pt-4 border-t border-gray-100 dark:border-gray-800 flex flex-col sm:flex-row justify-between gap-2 text-xs text-gray-400">
          <span>
            © {new Date().getFullYear()} {APP_NAME}. All rights reserved.
          </span>
          <span>v{APP_VERSION}</span>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
