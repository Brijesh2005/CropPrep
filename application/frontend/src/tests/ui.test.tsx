import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { ConfidenceGauge } from '@/components/prediction/ConfidenceGauge';
import { PredictionForm } from '@/components/prediction/PredictionForm';
import { FeatureChart } from '@/components/explainability/FeatureChart';

afterEach(cleanup);

describe('Button', () => {
  it('renders children and responds to clicks', () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Predict</Button>);
    const btn = screen.getByRole('button', { name: /predict/i });
    fireEvent.click(btn);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('disables while loading', () => {
    render(<Button loading>Go</Button>);
    expect(screen.getByRole('button', { name: /go/i })).toBeDisabled();
  });
});

describe('Badge', () => {
  it('renders with default neutral variant', () => {
    render(<Badge>Active</Badge>);
    const badge = screen.getByText('Active');
    expect(badge).toHaveClass('badge');
  });

  it('applies the success variant', () => {
    render(<Badge variant="success">Ready</Badge>);
    expect(screen.getByText('Ready')).toHaveClass('badge-success');
  });
});

describe('ConfidenceGauge', () => {
  it('renders a readable percentage', () => {
    render(<ConfidenceGauge value={0.85} />);
    expect(screen.getByText('85%')).toBeInTheDocument();
  });

  it('accepts a 0-100 value directly', () => {
    render(<ConfidenceGauge value={42} />);
    expect(screen.getByText('42%')).toBeInTheDocument();
  });
});

describe('PredictionForm', () => {
  it('submits parsed coordinates', () => {
    const onSubmit = vi.fn();
    render(<PredictionForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText(/latitude/i), {
      target: { value: '12.9141' },
    });
    fireEvent.change(screen.getByLabelText(/longitude/i), {
      target: { value: '74.8560' },
    });
    fireEvent.click(screen.getByRole('button', { name: /get prediction/i }));

    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ lat: 12.9141, lon: 74.856 }));
  });

  it('rejects invalid latitudes', () => {
    const onSubmit = vi.fn();
    render(<PredictionForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText(/latitude/i), {
      target: { value: '999' },
    });
    fireEvent.change(screen.getByLabelText(/longitude/i), {
      target: { value: '74.8' },
    });
    fireEvent.click(screen.getByRole('button', { name: /get prediction/i }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByText(/latitude must be between/i)).toBeInTheDocument();
  });
});

describe('FeatureChart', () => {
  it('shows a placeholder when there is no data', () => {
    render(<FeatureChart topFeatures={[]} />);
    expect(screen.getByText(/no feature contributions/i)).toBeInTheDocument();
  });
});
