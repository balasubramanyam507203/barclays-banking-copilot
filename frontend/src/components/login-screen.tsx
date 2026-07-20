"use client";

import {
  type FormEvent,
  useState,
} from "react";

import type {
  DevelopmentProfile,
} from "@/lib/types";

interface LoginScreenProps {
  isSubmitting: boolean;
  error: string | null;

  onLogin: (
    profile: DevelopmentProfile,
    password: string,
  ) => Promise<void>;
}

export default function LoginScreen({
  isSubmitting,
  error,
  onLogin,
}: LoginScreenProps) {
  const [profile, setProfile] =
    useState<DevelopmentProfile>(
      "compliance_analyst",
    );

  const [password, setPassword] =
    useState("");

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ): Promise<void> {
    event.preventDefault();

    await onLogin(
      profile,
      password,
    );
  }

  return (
    <main className="loginPage">
      <section className="loginCard">
        <div className="loginBrandMark">
          B
        </div>

        <p className="loginEyebrow">
          Enterprise Banking AI
        </p>

        <h1>Policy Copilot</h1>

        <p className="loginDescription">
          Sign in with a local development
          employee profile.
        </p>

        <form
          className="loginForm"
          onSubmit={handleSubmit}
        >
          <label htmlFor="employee-profile">
            Employee profile
          </label>

          <select
            id="employee-profile"
            value={profile}
            onChange={(event) => {
              setProfile(
                event.target
                  .value as DevelopmentProfile,
              );
            }}
            disabled={isSubmitting}
          >
            <option value="compliance_analyst">
              Compliance analyst
            </option>

            <option value="customer_support">
              Customer support
            </option>

            <option value="security_investigator">
              Security investigator
            </option>
          </select>

          <label htmlFor="development-password">
            Development password
          </label>

          <input
            id="development-password"
            type="password"
            value={password}
            onChange={(event) => {
              setPassword(
                event.target.value,
              );
            }}
            autoComplete="current-password"
            disabled={isSubmitting}
            required
          />

          {error !== null && (
            <div
              className="loginError"
              role="alert"
            >
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={
              isSubmitting ||
              password.length === 0
            }
          >
            {isSubmitting
              ? "Signing in…"
              : "Sign in"}
          </button>
        </form>

        <p className="loginNotice">
          This login screen is for local
          development. Amazon Cognito SSO and MFA
          replace it in production.
        </p>
      </section>
    </main>
  );
}