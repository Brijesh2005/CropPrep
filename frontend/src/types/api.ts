/** API types aligned with Phase 8 backend schemas. */

export type UserRole = 'farmer' | 'admin' | 'researcher' | 'user';

export type User = {
  id: number;
  email: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
};

export type AuthTokens = {
  access_token: string;
  refresh_token: string;
  token_type?: string;
  expires_in: number;
};

export type LoginRequest = {
  email: string;
  password: string;
};

export type RegisterRequest = {
  email: string;
  password: string;
  full_name?: string;
};

export type ProfileUpdateRequest = {
  full_name?: string;
  password?: string;
};

export type PredictionRequest = {
  lat: number;
  lon: number;
  year?: number;
  season?: string;
  include_explanation?: boolean;
};

export type MapPredictionRequest = {
  points: PredictionRequest[];
};

export type PredictionResponse = {
  prediction_id: number | null;
  location: Record<string, unknown>;
  coordinates: { lat?: number; lon?: number; [key: string]: unknown };
  recommended_crop: string;
  expected_yield: number | null;
  confidence: number;
  crop_probs: Record<string, number>;
  model_version: string;
  inference_time_ms: number;
  explanation_summary?: Record<string, unknown> | null;
  fallback?: boolean;
};

export type HistoryItem = {
  prediction_id: number;
  location: Record<string, unknown>;
  recommended_crop: string;
  expected_yield: number | null;
  confidence: number;
  model_version: string;
  created_at: string | null;
};

export type HistoryPage = {
  items: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
};

export type ExplanationResponse = {
  observation_id: string;
  crop: string;
  crop_probs: Record<string, number>;
  yield_prediction: number | null;
  confidence: Record<string, number> | number;
  top_features: Array<
    [string, number] | { name: string; contribution: number; value?: string | number }
  >;
  important_dates: string[];
  modality_gates: Record<string, number>;
  reasoning: string[];
  limitations: string[];
  raw: Record<string, unknown>;
};

export type ExplainRequest = {
  lat: number;
  lon: number;
  year?: number;
  season?: string;
};

export type LocationResponse = {
  id: string;
  lon: number;
  lat: number;
  name: string;
  admin: {
    village?: string;
    district?: string;
    state?: string;
    [key: string]: unknown;
  };
  distance_km?: number | null;
};

export type BoundaryResponse = {
  name: string;
  geometry_type: string;
  bbox: number[];
  features: number;
};

export type AdminDashboard = {
  model_ready: boolean;
  model_version: string;
  device: string;
  prediction_count: number;
  users_count: number;
  dataset_ready: boolean;
  queue_size: number;
};

export type AdminStatistics = {
  total_predictions: number;
  crop_distribution: Record<string, number>;
  avg_confidence: number;
  avg_inference_time_ms: number;
  fallback_count: number;
};

export type MonitoringMetrics = {
  requests: number;
  errors: number;
  avg_latency_ms: number;
  requests_per_second: number;
  uptime_seconds: number;
  by_path: Record<string, { requests: number; avg_ms: number; errors: number }>;
};

export type HealthResponse = {
  status: 'healthy' | 'degraded' | 'unhealthy' | string;
  version?: string;
  uptime?: number;
  services?: Record<string, boolean>;
};

export type ApiError = {
  code?: string;
  message?: string;
  detail?: string | Record<string, unknown>;
  trace_id?: string;
};
