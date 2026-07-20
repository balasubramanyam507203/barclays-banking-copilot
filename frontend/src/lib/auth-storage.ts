import type {
  CurrentUser,
} from "@/lib/types";

const ACCESS_TOKEN_KEY =
  "banking-copilot-access-token";

const CURRENT_USER_KEY =
  "banking-copilot-current-user";

export interface StoredAuthentication {
  accessToken: string;
  user: CurrentUser;
}

function browserStorageAvailable(): boolean {
  return typeof window !== "undefined";
}

export function saveAuthentication(
  authentication: StoredAuthentication,
): void {
  if (!browserStorageAvailable()) {
    return;
  }

  window.sessionStorage.setItem(
    ACCESS_TOKEN_KEY,
    authentication.accessToken,
  );

  window.sessionStorage.setItem(
    CURRENT_USER_KEY,
    JSON.stringify(authentication.user),
  );
}

export function getAccessToken():
  | string
  | null {
  if (!browserStorageAvailable()) {
    return null;
  }

  return window.sessionStorage.getItem(
    ACCESS_TOKEN_KEY,
  );
}

export function getStoredUser():
  | CurrentUser
  | null {
  if (!browserStorageAvailable()) {
    return null;
  }

  const rawUser =
    window.sessionStorage.getItem(
      CURRENT_USER_KEY,
    );

  if (rawUser === null) {
    return null;
  }

  try {
    return JSON.parse(
      rawUser,
    ) as CurrentUser;
  } catch {
    clearAuthentication();
    return null;
  }
}

export function clearAuthentication(): void {
  if (!browserStorageAvailable()) {
    return;
  }

  window.sessionStorage.removeItem(
    ACCESS_TOKEN_KEY,
  );

  window.sessionStorage.removeItem(
    CURRENT_USER_KEY,
  );
}