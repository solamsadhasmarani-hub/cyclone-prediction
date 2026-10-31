const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function fetchDashboard() {
  const response = await fetch(`${API_BASE_URL}/api/dashboard`);
  if (!response.ok) {
    throw new Error(`Failed to fetch dashboard data: ${response.status}`);
  }
  return response.json();
}

export async function analyzeSatelliteImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  const response = await fetch(`${API_BASE_URL}/api/analyze-image`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Image analysis failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error(`Failed to fetch health status: ${response.status}`);
  }
  return response.json();
}

export async function fetchModelStatus() {
  const response = await fetch(`${API_BASE_URL}/api/models/status`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model status: ${response.status}`);
  }
  return response.json();
}

export async function fetchCyclones() {
  const response = await fetch(`${API_BASE_URL}/api/cyclones`);
  if (!response.ok) {
    throw new Error(`Failed to fetch cyclones: ${response.status}`);
  }
  return response.json();
}

export async function fetchPredictionSummary() {
  const response = await fetch(`${API_BASE_URL}/api/predictions/summary`);
  if (!response.ok) {
    throw new Error(`Failed to fetch prediction summary: ${response.status}`);
  }
  return response.json();
}
