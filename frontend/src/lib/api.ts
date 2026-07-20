import {
  clearAuthentication,
  getAccessToken,
} from "@/lib/auth-storage";

import type {
  AccessTokenResponse,
  ApiErrorBody,
  ApiValidationError,
  ChatResponse,
  ConversationDetail,
  ConversationSummary,
  CurrentUser,
  DevelopmentProfile,
  FeedbackRating,
  FeedbackResponse,
  SourceDocument,
} from "@/lib/types";

const DEFAULT_API_BASE_URL =
  "http://127.0.0.1:8000/api/v1";

function getApiBaseUrl(): string {
  const configuredUrl =
    process.env
      .NEXT_PUBLIC_API_BASE_URL
      ?.trim();

  return (
    configuredUrl ||
    DEFAULT_API_BASE_URL
  ).replace(/\/+$/, "");
}

function formatValidationErrors(
  errors: ApiValidationError[],
): string {
  const messages = errors
    .map((error) => error.msg)
    .filter(
      (message): message is string =>
        typeof message === "string" &&
        message.length > 0,
    );

  return messages.length > 0
    ? messages.join(" ")
    : "The request was rejected by the API.";
}

async function getApiErrorMessage(
  response: Response,
): Promise<string> {
  try {
    const body =
      (await response.json()) as ApiErrorBody;

    if (typeof body.detail === "string") {
      return body.detail;
    }

    if (Array.isArray(body.detail)) {
      return formatValidationErrors(
        body.detail,
      );
    }
  } catch {
    // The response was not JSON.
  }

  if (response.status === 401) {
    return (
      "Your authentication session is invalid " +
      "or has expired."
    );
  }

  if (response.status === 404) {
    return "The requested item was not found.";
  }

  if (response.status === 503) {
    return (
      "The policy assistant is temporarily " +
      "unavailable."
    );
  }

  return (
    `The API request failed with status ` +
    `${response.status}.`
  );
}

function getAuthorizationHeaders():
  Record<string, string> {
  const token = getAccessToken();

  if (token === null) {
    throw new Error(
      "Sign in before using the policy assistant.",
    );
  }

  return {
    Authorization: `Bearer ${token}`,
  };
}

async function processAuthenticationStatus(
  response: Response,
): Promise<void> {
  if (response.status !== 401) {
    return;
  }

  clearAuthentication();

  if (typeof window !== "undefined") {
    window.dispatchEvent(
      new Event(
        "banking-copilot-auth-expired",
      ),
    );
  }
}

async function authenticatedFetch(
  url: string,
  options: RequestInit = {},
): Promise<Response> {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...getAuthorizationHeaders(),
      ...options.headers,
    },
  });

  await processAuthenticationStatus(
    response
  );

  return response;
}

export async function developmentLogin(
  profile: DevelopmentProfile,
  password: string,
): Promise<AccessTokenResponse> {
  const response = await fetch(
    `${getApiBaseUrl()}/auth/dev-login`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        profile,
        password,
      }),
    },
  );

  if (!response.ok) {
    throw new Error(
      await getApiErrorMessage(response),
    );
  }

  return (
    await response.json()
  ) as AccessTokenResponse;
}

export async function fetchCurrentUser():
  Promise<CurrentUser> {
  const response = await authenticatedFetch(
    `${getApiBaseUrl()}/auth/me`,
  );

  if (!response.ok) {
    throw new Error(
      await getApiErrorMessage(response),
    );
  }

  return (
    await response.json()
  ) as CurrentUser;
}

export async function askPolicyQuestion(
  question: string,
  conversationId: string | null,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await authenticatedFetch(
    `${getApiBaseUrl()}/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        conversation_id:
          conversationId,
      }),
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await getApiErrorMessage(response),
    );
  }

  return (
    await response.json()
  ) as ChatResponse;
}

export async function listConversations():
  Promise<ConversationSummary[]> {
  const response = await authenticatedFetch(
    `${getApiBaseUrl()}/conversations`,
  );

  if (!response.ok) {
    throw new Error(
      await getApiErrorMessage(response),
    );
  }

  return (
    await response.json()
  ) as ConversationSummary[];
}

export async function fetchConversation(
  conversationId: string,
  signal?: AbortSignal,
): Promise<ConversationDetail> {
  const encodedId =
    encodeURIComponent(conversationId);

  const response = await authenticatedFetch(
    `${getApiBaseUrl()}/conversations/${encodedId}`,
    {
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await getApiErrorMessage(response),
    );
  }

  return (
    await response.json()
  ) as ConversationDetail;
}

export async function submitFeedback(
  messageId: string,
  rating: FeedbackRating,
  comment?: string,
): Promise<FeedbackResponse> {
  const response = await authenticatedFetch(
    `${getApiBaseUrl()}/feedback`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message_id: messageId,
        rating,
        comment: comment || null,
      }),
    },
  );

  if (!response.ok) {
    throw new Error(
      await getApiErrorMessage(response),
    );
  }

  return (
    await response.json()
  ) as FeedbackResponse;
}

export async function fetchSourceDocument(
  documentId: string,
  signal?: AbortSignal,
): Promise<SourceDocument> {
  const encodedDocumentId =
    encodeURIComponent(documentId);

  const response = await authenticatedFetch(
    `${getApiBaseUrl()}/sources/` +
      encodedDocumentId,
    {
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await getApiErrorMessage(response),
    );
  }

  return (
    await response.json()
  ) as SourceDocument;
}