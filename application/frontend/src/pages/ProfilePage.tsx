import React, { useState } from 'react';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useAuth } from '@/hooks/useAuth';
import { authService } from '@/services/auth';
import { getErrorMessage } from '@/services/api';
import { useUiStore } from '@/store/uiStore';

export function ProfilePage() {
  const { user, refreshUser } = useAuth();
  const addToast = useUiStore((s) => s.addToast);

  const [fullName, setFullName] = useState(user?.full_name ?? '');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password && password.length < 8) {
      setError('New password must be at least 8 characters');
      return;
    }
    if (password !== confirm) {
      setError('Passwords do not match');
      return;
    }

    setSaving(true);
    try {
      await authService.updateProfile({
        full_name: fullName.trim() || undefined,
        password: password || undefined,
      });
      await refreshUser();
      setPassword('');
      setConfirm('');
      addToast('success', 'Profile updated');
    } catch (err) {
      setError(getErrorMessage(err, 'Could not update profile'));
    } finally {
      setSaving(false);
    }
  };

  if (!user) return null;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Profile</h1>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
          Manage your account details and password.
        </p>
      </div>

      <Card>
        <CardHeader className="flex items-center gap-4">
          <span className="flex h-14 w-14 items-center justify-center rounded-full bg-agriculture-100 text-agriculture-700 dark:bg-agriculture-900 dark:text-agriculture-300 text-xl font-bold">
            {(user.full_name || user.email || '?')[0].toUpperCase()}
          </span>
          <div>
            <CardTitle>{user.full_name || 'Unnamed user'}</CardTitle>
            <p className="text-sm text-gray-500">{user.email}</p>
            <div className="mt-1">
              <Badge variant={user.role === 'admin' ? 'warning' : 'success'}>{user.role}</Badge>
            </div>
          </div>
        </CardHeader>
        <CardBody>
          <form onSubmit={handleSubmit} className="max-w-lg space-y-4">
            <Input
              label="Full name"
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="Your full name"
            />
            <Input
              label="Email"
              type="email"
              value={user.email}
              disabled
              hint="Email cannot be changed."
            />
            <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
              <p className="mb-3 text-sm font-medium text-gray-700 dark:text-gray-200">
                Change password
              </p>
              <div className="space-y-4">
                <Input
                  label="New password"
                  type="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Leave blank to keep current password"
                />
                <Input
                  label="Confirm new password"
                  type="password"
                  autoComplete="new-password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder="Re-enter new password"
                />
              </div>
            </div>
            {error && (
              <p className="text-sm text-red-600" role="alert">
                {error}
              </p>
            )}
            <Button type="submit" loading={saving}>
              Save changes
            </Button>
          </form>
        </CardBody>
      </Card>
    </div>
  );
}

export default ProfilePage;
