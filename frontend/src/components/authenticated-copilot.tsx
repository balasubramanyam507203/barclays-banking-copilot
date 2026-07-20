"use client";

import {
  useEffect,
  useState,
} from "react";

import ChatInterface from
  "@/components/chat-interface";

import LoginScreen from
  "@/components/login-screen";

import {
  developmentLogin,
  fetchCurrentUser,
} from "@/lib/api";

import {
  clearAuthentication,
  getAccessToken,
  getStoredUser,
  saveAuthentication,
} from "@/lib/auth-storage";

import type {
  CurrentUser,
  DevelopmentProfile,
} from "@/lib/types";

export default function AuthenticatedCopilot() {
  const [user, setUser] =
    useState<CurrentUser | null>(null);

  const [isHydrating, setIsHydrating] =
    useState(true);

  const [isLoggingIn, setIsLoggingIn] =
    useState(false);

  const [loginError, setLoginError] =
    useState<string | null>(null);

  useEffect(() => {
    async function restoreSession():
      Promise<void> {
      const token = getAccessToken();
      const storedUser = getStoredUser();

      if (
        token === null ||
        storedUser === null
      ) {
        setIsHydrating(false);
        return;
      }

      try {
        const verifiedUser =
          await fetchCurrentUser();

        setUser(verifiedUser);
      } catch {
        clearAuthentication();
        setUser(null);
      } finally {
        setIsHydrating(false);
      }
    }

    void restoreSession();

    function handleExpiredSession(): void {
      clearAuthentication();
      setUser(null);
      setLoginError(
        "Your session expired. Sign in again.",
      );
    }

    window.addEventListener(
      "banking-copilot-auth-expired",
      handleExpiredSession,
    );

    return () => {
      window.removeEventListener(
        "banking-copilot-auth-expired",
        handleExpiredSession,
      );
    };
  }, []);

  async function handleLogin(
    profile: DevelopmentProfile,
    password: string,
  ): Promise<void> {
    setIsLoggingIn(true);
    setLoginError(null);

    try {
      const response =
        await developmentLogin(
          profile,
          password,
        );

      saveAuthentication({
        accessToken:
          response.access_token,
        user: response.user,
      });

      setUser(response.user);
    } catch (error) {
      setLoginError(
        error instanceof Error
          ? error.message
          : "Sign-in failed.",
      );
    } finally {
      setIsLoggingIn(false);
    }
  }

  function handleLogout(): void {
    clearAuthentication();
    setUser(null);
    setLoginError(null);
  }

  if (isHydrating) {
    return (
      <main className="sessionLoadingPage">
        <span className="spinner" />
        Verifying authentication session…
      </main>
    );
  }

  if (user === null) {
    return (
      <LoginScreen
        isSubmitting={isLoggingIn}
        error={loginError}
        onLogin={handleLogin}
      />
    );
  }

  return (
    <div className="authenticatedApplication">
      <ChatInterface />

      <aside className="authenticatedSession">
        <div>
          <strong>{user.username}</strong>

          <span>
            {user.role} · {user.region} ·
            Rank {user.clearance_rank}
          </span>
        </div>

        <button
          type="button"
          onClick={handleLogout}
        >
          Sign out
        </button>
      </aside>
    </div>
  );
}