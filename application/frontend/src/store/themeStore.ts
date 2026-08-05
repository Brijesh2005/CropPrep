import { create } from 'zustand';
import { THEME_STORAGE_KEY } from '@/config';
import type { Theme } from '@/types';

type ThemeState = {
  theme: Theme;
  resolved: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
  init: () => void;
};

function resolveTheme(theme: Theme): 'light' | 'dark' {
  if (theme === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  return theme;
}

function applyThemeClass(resolved: 'light' | 'dark') {
  const root = document.documentElement;
  root.classList.toggle('dark', resolved === 'dark');
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: 'system',
  resolved: 'light',

  setTheme: (theme) => {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
    const resolved = resolveTheme(theme);
    applyThemeClass(resolved);
    set({ theme, resolved });
  },

  init: () => {
    const saved = (localStorage.getItem(THEME_STORAGE_KEY) as Theme | null) ?? 'system';
    const resolved = resolveTheme(saved);
    applyThemeClass(resolved);
    set({ theme: saved, resolved });

    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => {
      if (get().theme === 'system') {
        const next = resolveTheme('system');
        applyThemeClass(next);
        set({ resolved: next });
      }
    };
    mq.addEventListener('change', onChange);
  },
}));
