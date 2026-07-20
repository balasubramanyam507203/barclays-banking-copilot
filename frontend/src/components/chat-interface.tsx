"use client";

import {
  type FormEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import ConversationHistory from
  "@/components/conversation-history";

import SourcePanel from
  "@/components/source-panel";

import {
  askPolicyQuestion,
  fetchConversation,
  fetchSourceDocument,
  listConversations,
  submitFeedback,
} from "@/lib/api";

import type {
  AssistantChatMessage,
  ChatMessage,
  ChatResponse,
  ConversationDetail,
  ConversationMessageRecord,
  ConversationSummary,
  FeedbackRating,
  SourceDocument,
  SourceReference,
  UserChatMessage,
} from "@/lib/types";

const EXAMPLE_QUESTIONS = [
  "What verification is required for high-risk international payments?",
  "What are the customer identity verification requirements?",
  "How should a customer complaint be handled?",
];

function createMessageId(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random()}`;
}

function formatTokenCount(
  value: number | undefined,
): string {
  if (typeof value !== "number") {
    return "Not reported";
  }

  return value.toLocaleString();
}

function buildHistoricalChatResponse(
  conversation: ConversationDetail,
  message: ConversationMessageRecord,
): ChatResponse {
  return {
    request_id:
      message.request_id || "",

    conversation_id:
      conversation.id,

    user_message_id: "",
    assistant_message_id:
      message.id,

    status:
      message.status || "answered",

    answer: message.content,
    abstained: message.abstained,
    model_called: message.model_called,

    citations_used:
      message.citations_used,

    sources: message.sources,

    evidence_count:
      message.evidence_count,

    guardrails:
      message.guardrails || {
        citation_validation_passed: true,
        post_generation_guardrails_passed:
          true,
        claims_checked: 0,
        supported_claims: 0,
      },

    usage:
      message.usage || {},
  };
}

function convertConversationMessages(
  conversation: ConversationDetail,
): ChatMessage[] {
  return conversation.messages.map(
    (message): ChatMessage => {
      if (message.role === "user") {
        return {
          id: message.id,
          role: "user",
          content: message.content,
        };
      }

      return {
        id: message.id,
        role: "assistant",
        content: message.content,
        response:
          buildHistoricalChatResponse(
            conversation,
            message,
          ),
        feedback:
          message.feedback || null,
      };
    },
  );
}

export default function ChatInterface() {
  const [question, setQuestion] =
    useState("");

  const [messages, setMessages] =
    useState<ChatMessage[]>([]);

  const [
    activeConversationId,
    setActiveConversationId,
  ] = useState<string | null>(null);

  const [
    conversations,
    setConversations,
  ] = useState<ConversationSummary[]>([]);

  const [
    isHistoryLoading,
    setIsHistoryLoading,
  ] = useState(true);

  const [
    historyError,
    setHistoryError,
  ] = useState<string | null>(null);

  const [
    isConversationLoading,
    setIsConversationLoading,
  ] = useState(false);

  const [isSubmitting, setIsSubmitting] =
    useState(false);

  const [requestError, setRequestError] =
    useState<string | null>(null);

  const [feedbackError, setFeedbackError] =
    useState<string | null>(null);

  const [
    submittingFeedbackMessageId,
    setSubmittingFeedbackMessageId,
  ] = useState<string | null>(null);

  const [selectedSource, setSelectedSource] =
    useState<SourceReference | null>(null);

  const [sourceDocument, setSourceDocument] =
    useState<SourceDocument | null>(null);

  const [sourceError, setSourceError] =
    useState<string | null>(null);

  const [isSourceLoading, setIsSourceLoading] =
    useState(false);

  const conversationEndRef =
    useRef<HTMLDivElement | null>(null);

  const sourceAbortControllerRef =
    useRef<AbortController | null>(null);

  const conversationAbortControllerRef =
    useRef<AbortController | null>(null);

  const loadConversationHistory =
    useCallback(async (): Promise<void> => {
      setIsHistoryLoading(true);
      setHistoryError(null);

      try {
        const history =
          await listConversations();

        setConversations(history);
      } catch (error) {
        setHistoryError(
          error instanceof Error
            ? error.message
            : "Conversation history failed to load.",
        );
      } finally {
        setIsHistoryLoading(false);
      }
    }, []);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void loadConversationHistory();
    }, 0);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [loadConversationHistory]);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [
    messages,
    isSubmitting,
  ]);

  useEffect(() => {
    return () => {
      sourceAbortControllerRef.current?.abort();
      conversationAbortControllerRef
        .current
        ?.abort();
    };
  }, []);

  async function selectConversation(
    conversationId: string,
  ): Promise<void> {
    if (
      isSubmitting ||
      isConversationLoading
    ) {
      return;
    }

    conversationAbortControllerRef
      .current
      ?.abort();

    const controller =
      new AbortController();

    conversationAbortControllerRef.current =
      controller;

    setActiveConversationId(
      conversationId
    );

    setIsConversationLoading(true);
    setRequestError(null);
    setFeedbackError(null);

    try {
      const conversation =
        await fetchConversation(
          conversationId,
          controller.signal,
        );

      setMessages(
        convertConversationMessages(
          conversation
        )
      );
    } catch (error) {
      if (
        error instanceof DOMException &&
        error.name === "AbortError"
      ) {
        return;
      }

      setRequestError(
        error instanceof Error
          ? error.message
          : "Conversation failed to load.",
      );
    } finally {
      if (
        conversationAbortControllerRef
          .current === controller
      ) {
        setIsConversationLoading(false);
      }
    }
  }

  function startNewConversation(): void {
    conversationAbortControllerRef
      .current
      ?.abort();

    setActiveConversationId(null);
    setMessages([]);
    setQuestion("");
    setRequestError(null);
    setFeedbackError(null);
  }

  async function submitQuestion(
    submittedQuestion: string,
  ): Promise<void> {
    const normalizedQuestion =
      submittedQuestion.trim();

    if (
      !normalizedQuestion ||
      isSubmitting
    ) {
      return;
    }

    const optimisticMessageId =
      createMessageId();

    const userMessage: UserChatMessage = {
      id: optimisticMessageId,
      role: "user",
      content: normalizedQuestion,
    };

    setMessages((currentMessages) => [
      ...currentMessages,
      userMessage,
    ]);

    setQuestion("");
    setRequestError(null);
    setFeedbackError(null);
    setIsSubmitting(true);

    try {
      const response =
        await askPolicyQuestion(
          normalizedQuestion,
          activeConversationId,
        );

      setActiveConversationId(
        response.conversation_id
      );

      setMessages((currentMessages) => {
        const updatedMessages =
          currentMessages.map(
            (message) =>
              message.id ===
              optimisticMessageId
                ? {
                    ...message,
                    id:
                      response.user_message_id,
                  }
                : message,
          );

        const assistantMessage:
          AssistantChatMessage = {
          id:
            response.assistant_message_id,
          role: "assistant",
          content: response.answer,
          response,
          feedback: null,
        };

        return [
          ...updatedMessages,
          assistantMessage,
        ];
      });

      await loadConversationHistory();
    } catch (error) {
      setRequestError(
        error instanceof Error
          ? error.message
          : "The policy request failed.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ): Promise<void> {
    event.preventDefault();

    await submitQuestion(question);
  }

  function handleQuestionKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>,
  ): void {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      void submitQuestion(question);
    }
  }

  async function handleFeedback(
    messageId: string,
    rating: FeedbackRating,
  ): Promise<void> {
    setSubmittingFeedbackMessageId(
      messageId
    );

    setFeedbackError(null);

    try {
      const feedback =
        await submitFeedback(
          messageId,
          rating,
        );

      setMessages((currentMessages) =>
        currentMessages.map((message) => {
          if (
            message.role === "assistant" &&
            message.id === messageId
          ) {
            return {
              ...message,
              feedback,
            };
          }

          return message;
        }),
      );
    } catch (error) {
      setFeedbackError(
        error instanceof Error
          ? error.message
          : "Feedback could not be saved.",
      );
    } finally {
      setSubmittingFeedbackMessageId(null);
    }
  }

  async function openSource(
    source: SourceReference,
  ): Promise<void> {
    sourceAbortControllerRef.current?.abort();

    const controller =
      new AbortController();

    sourceAbortControllerRef.current =
      controller;

    setSelectedSource(source);
    setSourceDocument(null);
    setSourceError(null);
    setIsSourceLoading(true);

    try {
      const document =
        await fetchSourceDocument(
          source.document_id,
          controller.signal,
        );

      setSourceDocument(document);
    } catch (error) {
      if (
        error instanceof DOMException &&
        error.name === "AbortError"
      ) {
        return;
      }

      setSourceError(
        error instanceof Error
          ? error.message
          : "The source could not be loaded.",
      );
    } finally {
      if (
        sourceAbortControllerRef.current ===
        controller
      ) {
        setIsSourceLoading(false);
      }
    }
  }

  function closeSourcePanel(): void {
    sourceAbortControllerRef.current?.abort();
    sourceAbortControllerRef.current = null;

    setSelectedSource(null);
    setSourceDocument(null);
    setSourceError(null);
    setIsSourceLoading(false);
  }

  const hasMessages =
    messages.length > 0;

  return (
    <>
      <section className="copilotShell">
        <aside className="sidebar">
          <div className="brand">
            <div
              className="brandMark"
              aria-hidden="true"
            >
              B
            </div>

            <div>
              <p className="brandEyebrow">
                Enterprise AI
              </p>

              <h1>Policy Copilot</h1>
            </div>
          </div>

          <ConversationHistory
            conversations={conversations}
            activeConversationId={
              activeConversationId
            }
            isLoading={isHistoryLoading}
            error={historyError}
            onNewConversation={
              startNewConversation
            }
            onSelectConversation={(
              conversationId,
            ) => {
              void selectConversation(
                conversationId
              );
            }}
          />
        </aside>

        <main className="chatWorkspace">
          <header className="workspaceHeader">
            <div>
              <p className="workspaceEyebrow">
                Banking policy and compliance
              </p>

              <h2>
                Ask an authorized policy question
              </h2>
            </div>

            <div className="securityBadge">
              <span aria-hidden="true">✓</span>
              Guardrails enabled
            </div>
          </header>

          <div className="conversation">
            {isConversationLoading && (
              <div className="conversationLoading">
                <span className="spinner" />
                Loading conversation…
              </div>
            )}

            {!hasMessages &&
              !isConversationLoading && (
                <section className="welcomeState">
                  <div
                    className="welcomeIcon"
                    aria-hidden="true"
                  >
                    B
                  </div>

                  <h2>
                    How can I help with policy today?
                  </h2>

                  <p>
                    Ask about identity verification,
                    international payments,
                    complaints, or another indexed
                    banking policy.
                  </p>

                  <div className="exampleQuestions">
                    {EXAMPLE_QUESTIONS.map(
                      (exampleQuestion) => (
                        <button
                          type="button"
                          key={exampleQuestion}
                          onClick={() => {
                            void submitQuestion(
                              exampleQuestion
                            );
                          }}
                          disabled={isSubmitting}
                        >
                          {exampleQuestion}
                        </button>
                      ),
                    )}
                  </div>
                </section>
              )}

            {messages.map((message) => {
              if (message.role === "user") {
                return (
                  <article
                    className="messageRow userMessageRow"
                    key={message.id}
                  >
                    <div className="messageAvatar userAvatar">
                      You
                    </div>

                    <div className="messageBubble userBubble">
                      <p>{message.content}</p>
                    </div>
                  </article>
                );
              }

              const response =
                message.response;

              const feedbackSubmitting =
                submittingFeedbackMessageId ===
                message.id;

              return (
                <article
                  className="messageRow assistantMessageRow"
                  key={message.id}
                >
                  <div className="messageAvatar assistantAvatar">
                    AI
                  </div>

                  <div className="assistantResponse">
                    <div className="messageBubble assistantBubble">
                      <div className="answerHeader">
                        <strong>
                          Policy Copilot
                        </strong>

                        <span
                          className={
                            response.abstained
                              ? "answerStatus abstainedStatus"
                              : "answerStatus answeredStatus"
                          }
                        >
                          {response.abstained
                            ? "Evidence unavailable"
                            : "Guardrail approved"}
                        </span>
                      </div>

                      <p className="answerText">
                        {message.content}
                      </p>
                    </div>

                    {!response.abstained && (
                      <div className="feedbackControls">
                        <span>
                          Was this answer helpful?
                        </span>

                        <button
                          type="button"
                          className={
                            message.feedback?.rating ===
                            "helpful"
                              ? "selectedFeedback"
                              : ""
                          }
                          disabled={
                            feedbackSubmitting
                          }
                          onClick={() => {
                            void handleFeedback(
                              message.id,
                              "helpful",
                            );
                          }}
                        >
                          👍 Helpful
                        </button>

                        <button
                          type="button"
                          className={
                            message.feedback?.rating ===
                            "not_helpful"
                              ? "selectedFeedback"
                              : ""
                          }
                          disabled={
                            feedbackSubmitting
                          }
                          onClick={() => {
                            void handleFeedback(
                              message.id,
                              "not_helpful",
                            );
                          }}
                        >
                          👎 Not helpful
                        </button>
                      </div>
                    )}

                    {response.sources.length >
                      0 && (
                      <section className="citationSection">
                        <h3>
                          Authorized sources
                        </h3>

                        <div className="citationGrid">
                          {response.sources.map(
                            (source) => (
                              <button
                                type="button"
                                className="citationCard"
                                key={
                                  `${message.id}-` +
                                  source.chunk_id
                                }
                                onClick={() => {
                                  void openSource(
                                    source
                                  );
                                }}
                              >
                                <span className="citationLabel">
                                  {source.label}
                                </span>

                                <strong>
                                  {source.title}
                                </strong>

                                <span>
                                  {
                                    source.document_id
                                  }{" "}
                                  · Version{" "}
                                  {source.version}
                                </span>

                                <span className="viewSource">
                                  View evidence →
                                </span>
                              </button>
                            ),
                          )}
                        </div>
                      </section>
                    )}

                    <details className="diagnostics">
                      <summary>
                        Answer diagnostics
                      </summary>

                      <dl>
                        <div>
                          <dt>Request ID</dt>
                          <dd>
                            {response.request_id ||
                              "Not reported"}
                          </dd>
                        </div>

                        <div>
                          <dt>
                            Citation validation
                          </dt>
                          <dd>
                            {response.guardrails
                              .citation_validation_passed
                              ? "Passed"
                              : "Failed"}
                          </dd>
                        </div>

                        <div>
                          <dt>
                            Claim guardrails
                          </dt>
                          <dd>
                            {response.guardrails
                              .post_generation_guardrails_passed
                              ? "Passed"
                              : "Failed"}
                          </dd>
                        </div>

                        <div>
                          <dt>
                            Supported claims
                          </dt>
                          <dd>
                            {
                              response.guardrails
                                .supported_claims
                            }
                            /
                            {
                              response.guardrails
                                .claims_checked
                            }
                          </dd>
                        </div>

                        <div>
                          <dt>Total tokens</dt>
                          <dd>
                            {formatTokenCount(
                              response.usage
                                .total_tokens,
                            )}
                          </dd>
                        </div>
                      </dl>
                    </details>
                  </div>
                </article>
              );
            })}

            {isSubmitting && (
              <article className="messageRow assistantMessageRow">
                <div className="messageAvatar assistantAvatar">
                  AI
                </div>

                <div className="messageBubble loadingBubble">
                  <span className="spinner" />

                  <div>
                    <strong>
                      Reviewing authorized policies
                    </strong>

                    <p>
                      Retrieving, reranking and
                      validating evidence…
                    </p>
                  </div>
                </div>
              </article>
            )}

            {requestError !== null && (
              <div
                className="errorBanner"
                role="alert"
              >
                <strong>
                  The request could not be completed.
                </strong>

                <p>{requestError}</p>
              </div>
            )}

            {feedbackError !== null && (
              <div
                className="errorBanner"
                role="alert"
              >
                <strong>
                  Feedback was not saved.
                </strong>

                <p>{feedbackError}</p>
              </div>
            )}

            <div ref={conversationEndRef} />
          </div>

          <footer className="composerArea">
            <form
              className="composer"
              onSubmit={handleSubmit}
            >
              <label
                className="srOnly"
                htmlFor="policy-question"
              >
                Policy question
              </label>

              <textarea
                id="policy-question"
                value={question}
                onChange={(event) =>
                  setQuestion(
                    event.target.value
                  )
                }
                onKeyDown={
                  handleQuestionKeyDown
                }
                placeholder="Ask a banking policy question…"
                rows={2}
                maxLength={2000}
                disabled={
                  isSubmitting ||
                  isConversationLoading
                }
              />

              <div className="composerActions">
                <span>
                  Enter to send · Shift+Enter for a
                  new line
                </span>

                <button
                  type="submit"
                  disabled={
                    isSubmitting ||
                    isConversationLoading ||
                    question.trim().length === 0
                  }
                >
                  {isSubmitting
                    ? "Reviewing…"
                    : "Send"}
                </button>
              </div>
            </form>

            <p className="composerNotice">
              Answers are generated only from
              authorized indexed evidence. Verify
              critical decisions with the policy
              owner.
            </p>
          </footer>
        </main>
      </section>

      <SourcePanel
        source={selectedSource}
        document={sourceDocument}
        isLoading={isSourceLoading}
        error={sourceError}
        onClose={closeSourcePanel}
      />
    </>
  );
}