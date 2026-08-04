import { create } from 'zustand';
import { getStoredTokens } from '@/services/api';
import type { User } from '@/types';

type AuthState = {
  user: User | null;
  tokensPresent: boolean;
  setUser: (user: User | null) => void;
  setTokensPresent: (present: boolean) => void;
  syncTokens: () => void;
  clear: () => void;
};

/**
 * Lightweight auth store for components that need token presence without
 * triggering an API round-trip. The authoritative auth state lives in
 * `AuthContext`; this store mirrors it for convenience.
 */
export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  tokensPresent: getStoredTokens() !== null,

  setUser: (user) => set({ user }),
  setTokensPresent: (tokensPresent) => set({ tokensPresent }),
  syncTokens: () => set({ tokensPresent: getStoredTokens() !== null }),
  clear: () => set({ user: null, tokensPresent: false }),
}));

export default useAuthStore;
