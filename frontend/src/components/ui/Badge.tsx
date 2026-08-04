import React from 'react';
import { cn } from '@/utils/cn';

type BadgeVariant = 'success' | 'warning' | 'danger' | 'info' | 'neutral';

type BadgeProps = {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
};

const variantClass: Record<BadgeVariant, string> = {
  success: 'badge-success',
  warning: 'badge-warning',
  danger: 'badge-danger',
  info: 'badge-info',
  neutral: 'badge bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
};

export function Badge({ children, variant = 'neutral', className }: BadgeProps) {
  return <span className={cn(variantClass[variant], className)}>{children}</span>;
}

export default Badge;
