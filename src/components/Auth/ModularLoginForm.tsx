import { useEffect, useState } from 'react';
import { apiClient } from '../../api/apiClient';
import { useAuthStore } from '../../store/authStore';
import LoginHeader from './LoginHeader';
import EmailPasswordForm from './EmailPasswordForm';
import SocialLoginButtons from './SocialLoginButtons';
import Divider from './Divider';
import PasskeyPanel from './PasskeyPanel';
import TurnstileWidget from '../TurnstileWidget';
import { useTurnstile } from '../../config/turnstile';

interface ModularLoginFormProps {
  onSuccess: () => void;
  // eslint-disable-next-line no-unused-vars
  onError: (message: string) => void;
  initialMode?: 'login' | 'register';
}

export default function ModularLoginForm({
  onSuccess,
  onError,
  initialMode = 'login',
}: ModularLoginFormProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [isRegister, setIsRegister] = useState(initialMode === 'register');
  const [showPasskey, setShowPasskey] = useState(false);
  const [email, setEmail] = useState('');
  const [turnstileToken, setTurnstileToken] = useState('');

  const turnstileConfig = useTurnstile('login');

  useEffect(() => {
    setIsRegister(initialMode === 'register');
  }, [initialMode]);

  const handleEmailPasswordSubmit = async (email: string, password: string) => {
    setEmail(email); // Store for passkey

    // Verify Turnstile CAPTCHA completion (security requirement)
    if (!turnstileToken) {
      onError('Please complete the security verification');
      return;
    }

    setIsLoading(true);

    try {
      const response = isRegister
        ? await apiClient.register(email, password, turnstileToken)
        : await apiClient.login(email, password);

      const tokenValue = (response as any).access_token || (response as any).token || null;
      if (!tokenValue) {
        throw new Error('Authentication failed - invalid server response');
      }
      const resolvedUser = (response as any).user ?? { id: email, email };
      useAuthStore.getState().setSession({
        token: tokenValue,
        user: resolvedUser,
        expiresIn: (response as any).expires_in,
      });
      onSuccess();
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Authentication failed');
      // Reset Turnstile on error
      setTurnstileToken('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      const { url } = (await apiClient.getGoogleAuthUrl()) as { url: string };
      window.location.href = url;
    } catch (error) {
      onError('Failed to initiate Google login');
    }
  };

  return (
    <div className="w-full max-w-md">
      <div className="bg-surface rounded-2xl shadow-[0_20px_60px_rgba(15,23,42,0.12)] border border-border p-8">
        <LoginHeader isRegister={isRegister} />

        <EmailPasswordForm
          onSubmit={handleEmailPasswordSubmit}
          isRegister={isRegister}
          isLoading={isLoading}
        />

        {/* Turnstile Bot Protection */}
        <div className="mt-4">
          <TurnstileWidget
            siteKey={turnstileConfig.siteKey}
            onVerify={token => setTurnstileToken(token)}
            mode="managed"
            theme="auto"
            size="normal"
            onError={error => {
              console.error('Turnstile verification failed:', error);
              onError('Security verification failed. Please try again.');
            }}
          />
        </div>

        <Divider text="Or continue with" />

        <SocialLoginButtons onGoogleLogin={handleGoogleLogin} isLoading={isLoading} />

        <div className="mt-6 space-y-3">
          <button
            onClick={() => setIsRegister(!isRegister)}
            className="w-full text-center text-primary hover:text-primary-600 text-sm font-medium transition-colors"
            type="button"
          >
            {isRegister ? 'Already have an account? Sign in' : "Don't have an account? Sign up"}
          </button>

          <button
            onClick={() => setShowPasskey(!showPasskey)}
            className="w-full text-center text-accent hover:text-accent-600 text-sm font-medium transition-colors"
            type="button"
          >
            {showPasskey ? 'Hide Passkey Options' : '🔐 Use Passkey (WebAuthn)'}
          </button>
        </div>

        {showPasskey && (
          <div className="mt-6 pt-6 border-t border-divider">
            <PasskeyPanel email={email} onError={onError} onSuccess={onSuccess} />
          </div>
        )}
      </div>

      <p className="text-xs text-center text-muted mt-6">
        By signing in you agree to anonymous usage data for quality and reliability.
      </p>
    </div>
  );
}
