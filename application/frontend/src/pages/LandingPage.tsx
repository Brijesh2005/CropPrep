import React from 'react';
import { Link } from 'react-router-dom';
import { Header } from '@/components/layout/Header';
import { Footer } from '@/components/layout/Footer';
import { Button } from '@/components/ui/Button';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { APP_NAME, APP_DESCRIPTION } from '@/config';
import { useAuth } from '@/hooks/useAuth';

const features = [
  {
    icon: '📍',
    title: 'Zero-effort predictions',
    description:
      'No manual soil tests or data entry. Select a point on the map or use your GPS location.',
  },
  {
    icon: '🌾',
    title: 'Multi-crop recommendation',
    description:
      'Hybrid TabTransformer + CNN + Temporal Transformer fuses satellite imagery and structured data.',
  },
  {
    icon: '📈',
    title: 'Yield forecasting',
    description:
      'Predict expected yield per hectare using 8 years of multi-temporal Sentinel-2 vegetation indices.',
  },
  {
    icon: '🧠',
    title: 'Explainable AI',
    description:
      'SHAP values, attention maps and GradCAM heatmaps show exactly why a crop was recommended.',
  },
  {
    icon: '🗺️',
    title: 'Spatio-temporal alignment',
    description:
      'The STAM module aligns any GPS point with its complete agricultural context for any season.',
  },
  {
    icon: '🌦️',
    title: 'Season-aware insights',
    description:
      'Kharif, Rabi and Zaid crop calendars with growing-window-aware sampling built in.',
  },
];

const steps = [
  {
    title: 'Open the map',
    description: 'The interactive map shows all dataset-covered villages across Dakshina Kannada.',
  },
  {
    title: 'Pick a location',
    description: 'Click anywhere, search a village, or use the “My location” button (GPS).',
  },
  {
    title: 'Get the prediction',
    description:
      'The system returns the recommended crop, expected yield, confidence and a full explanation.',
  },
];

export function LandingPage() {
  const { isAuthenticated } = useAuth();

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-950">
      <Header />

      <main className="flex-1">
        {/* Hero */}
        <section className="relative overflow-hidden bg-gradient-to-br from-agriculture-50 via-white to-agriculture-100 dark:from-gray-900 dark:via-gray-950 dark:to-agriculture-950">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 py-20 sm:py-28 text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-agriculture-200 dark:border-agriculture-800 bg-agriculture-50 dark:bg-agriculture-900/40 px-4 py-1 text-xs font-medium text-agriculture-800 dark:text-agriculture-300">
              AI-Powered Agricultural Decision Support System
            </span>
            <h1 className="mt-6 text-4xl sm:text-6xl font-bold tracking-tight text-gray-900 dark:text-white text-balance">
              Grow the right crop,{' '}
              <span className="text-agriculture-600 dark:text-agriculture-400">
                right where you are.
              </span>
            </h1>
            <p className="mx-auto mt-6 max-w-2xl text-lg text-gray-600 dark:text-gray-300">
              {APP_DESCRIPTION}. {APP_NAME} fuses satellite imagery with structured agricultural
              data to recommend the best crop and forecast your yield — from a single map click.
            </p>
            <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to={isAuthenticated ? '/map' : '/register'}>
                <Button size="lg">Start a prediction →</Button>
              </Link>
              <Link to={isAuthenticated ? '/dashboard' : '/login'}>
                <Button variant="outline" size="lg">
                  {isAuthenticated ? 'View dashboard' : 'Sign in'}
                </Button>
              </Link>
            </div>
          </div>
        </section>

        {/* How it works */}
        <section className="mx-auto max-w-7xl px-4 sm:px-6 py-16">
          <h2 className="text-center text-3xl font-bold text-gray-900 dark:text-white">
            How it works
          </h2>
          <p className="mt-2 text-center text-gray-600 dark:text-gray-300">
            From GPS coordinate to recommendation in three steps.
          </p>
          <div className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-6">
            {steps.map((step, i) => (
              <Card key={step.title} className="relative">
                <span className="absolute top-4 right-4 flex h-8 w-8 items-center justify-center rounded-full bg-agriculture-100 text-agriculture-700 dark:bg-agriculture-900/50 dark:text-agriculture-300 font-bold">
                  {i + 1}
                </span>
                <CardHeader>
                  <CardTitle>{step.title}</CardTitle>
                </CardHeader>
                <CardBody>
                  <p className="text-sm text-gray-600 dark:text-gray-300">{step.description}</p>
                </CardBody>
              </Card>
            ))}
          </div>
        </section>

        {/* Features */}
        <section className="bg-white dark:bg-gray-900 border-y border-gray-200 dark:border-gray-800">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 py-16">
            <h2 className="text-center text-3xl font-bold text-gray-900 dark:text-white">
              Built on research-grade AI
            </h2>
            <p className="mt-2 text-center text-gray-600 dark:text-gray-300">
              A hybrid machine-learning / deep-learning spatio-temporal cross-modal fusion
              framework.
            </p>
            <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {features.map((f) => (
                <Card key={f.title}>
                  <CardBody>
                    <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-agriculture-100 dark:bg-agriculture-900/40 text-2xl">
                      {f.icon}
                    </span>
                    <h3 className="mt-4 font-semibold text-gray-900 dark:text-white">{f.title}</h3>
                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">{f.description}</p>
                  </CardBody>
                </Card>
              ))}
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="mx-auto max-w-7xl px-4 sm:px-6 py-16">
          <div className="rounded-2xl bg-agriculture-600 px-8 py-12 text-center shadow-lg">
            <h2 className="text-2xl sm:text-3xl font-bold text-white">
              Ready to see what your land can grow?
            </h2>
            <p className="mt-3 text-agriculture-50">
              Free for farmers in Dakshina Kannada. No data entry required.
            </p>
            <div className="mt-8">
              <Link to={isAuthenticated ? '/map' : '/register'}>
                <Button size="lg" className="bg-white text-agriculture-700 hover:bg-agriculture-50">
                  Get started free
                </Button>
              </Link>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}

export default LandingPage;
