import api, { storeTokens, clearTokens, getStoredTokens } from './api';
import type {
  AuthTokens,
  LoginRequest,
  RegisterRequest,
  User,
  ProfileUpdateRequest,
} from '@/types';

export const authService = {
  async login(credentials: LoginRequest): Promise<{ tokens: AuthTokens; user: User }> {
    const formData = new URLSearchParams();
    formData.append('username', credentials.email);
    formData.append('password', credentials.password);

    const response = await api.post<AuthTokens>('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    storeTokens(response.data);
    const user = await this.getCurrentUser();
    return { tokens: response.data, user };
  },

  async register(data: RegisterRequest): Promise<User> {
    const response = await api.post<User>('/auth/register', data);
    return response.data;
  },

  async logout(): Promise<void> {
    try {
      await api.post('/auth/logout');
    } catch {
      // Ignore logout errors (token may already be invalid)
    } finally {
      clearTokens();
    }
  },

  async getCurrentUser(): Promise<User> {
    const response = await api.get<User>('/users/me');
    return response.data;
  },

  async updateProfile(data: ProfileUpdateRequest): Promise<User> {
    const response = await api.put<User>('/users/profile', data);
    return response.data;
  },

  async refreshToken(refreshToken: string): Promise<AuthTokens> {
    const response = await api.post<AuthTokens>('/auth/refresh', {
      refresh_token: refreshToken,
    });
    storeTokens(response.data);
    return response.data;
  },

  isAuthenticated(): boolean {
    return getStoredTokens() !== null;
  },
};
