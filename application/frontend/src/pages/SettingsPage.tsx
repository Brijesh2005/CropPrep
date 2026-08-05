import React from 'react';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { useTheme } from '@/contexts/ThemeContext';
import { useUiStore } from '@/store/uiStore';
import type { Theme } from '@/types';

const themes: Array<{ value: Theme; label: string }> = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'system', label: 'System' },
];

export function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const addToast = useUiStore((s) => s.addToast);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
          Customize your CropFusion experience.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Appearance</CardTitle>
        </CardHeader>
        <CardBody>
          <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
            Choose how CropFusion looks on your device.
          </p>
          <div className="flex gap-2">
            {themes.map((t) => (
              <Button
                key={t.value}
                variant={theme === t.value ? 'primary' : 'outline'}
                onClick={() => setTheme(t.value)}
              >
                {t.label}
              </Button>
            ))}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Notifications</CardTitle>
        </CardHeader>
        <CardBody>
          <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
            Predictions are saved to your history automatically. Browser notifications are not
            currently used.
          </p>
          <Button
            variant="outline"
            onClick={() => addToast('info', 'Notifications preferences saved')}
          >
            Save preferences
          </Button>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Data & privacy</CardTitle>
        </CardHeader>
        <CardBody>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Your prediction history is stored securely on the server and linked to your account. Use
            the History page to review or export it.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}

export default SettingsPage;
