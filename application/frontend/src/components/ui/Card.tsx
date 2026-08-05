import React from 'react';
import { cn } from '@/utils/cn';

type DivProps = React.HTMLAttributes<HTMLDivElement>;

export function Card({ className, children, ...rest }: DivProps) {
  return (
    <div className={cn('card', className)} {...rest}>
      {children}
    </div>
  );
}

export function CardHeader({ className, children, ...rest }: DivProps) {
  return (
    <div className={cn('card-header', className)} {...rest}>
      {children}
    </div>
  );
}

export function CardBody({ className, children, ...rest }: DivProps) {
  return (
    <div className={cn('card-body', className)} {...rest}>
      {children}
    </div>
  );
}

export function CardFooter({ className, children, ...rest }: DivProps) {
  return (
    <div className={cn('card-footer', className)} {...rest}>
      {children}
    </div>
  );
}

export function CardTitle({
  className,
  children,
  ...rest
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3 className={cn('text-lg font-semibold text-gray-900 dark:text-white', className)} {...rest}>
      {children}
    </h3>
  );
}

export default Card;
