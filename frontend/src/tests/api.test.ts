import { describe, it, expect } from 'vitest';
import axios from 'axios';
import { getErrorMessage } from '@/services/api';

describe('getErrorMessage', () => {
  it('extracts a string detail from an axios error', () => {
    const error = new axios.AxiosError('Request failed');
    error.response = {
      data: { detail: 'Invalid credentials' },
      status: 401,
      statusText: 'Unauthorized',
      headers: {},
      config: {} as never,
    };
    expect(getErrorMessage(error, 'fallback')).toBe('Invalid credentials');
  });

  it('extracts a nested detail.message', () => {
    const error = new axios.AxiosError('Request failed');
    error.response = {
      data: { detail: { message: 'Rate limited' } },
      status: 429,
      statusText: 'Too Many Requests',
      headers: {},
      config: {} as never,
    };
    expect(getErrorMessage(error)).toBe('Rate limited');
  });

  it('uses the fallback for unknown errors', () => {
    expect(getErrorMessage('nope', 'Something went wrong')).toBe('Something went wrong');
  });
});
