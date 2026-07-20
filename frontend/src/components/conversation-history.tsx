"use client";

import type {
  ConversationSummary,
} from "@/lib/types";

interface ConversationHistoryProps {
  conversations: ConversationSummary[];
  activeConversationId: string | null;

  isLoading: boolean;
  error: string | null;

  onNewConversation: () => void;
  onSelectConversation: (
    conversationId: string,
  ) => void;
}

function formatConversationDate(
  value: string,
): string {
  const date = new Date(value);

  return new Intl.DateTimeFormat(
    undefined,
    {
      month: "short",
      day: "numeric",
    },
  ).format(date);
}

export default function ConversationHistory({
  conversations,
  activeConversationId,
  isLoading,
  error,
  onNewConversation,
  onSelectConversation,
}: ConversationHistoryProps) {
  return (
    <section className="historySection">
      <button
        type="button"
        className="newConversationButton"
        onClick={onNewConversation}
      >
        <span aria-hidden="true">＋</span>
        New conversation
      </button>

      <div className="historyHeading">
        <span>Conversation history</span>
      </div>

      {isLoading && (
        <div className="historyLoading">
          Loading history…
        </div>
      )}

      {error !== null && (
        <div className="historyError">
          {error}
        </div>
      )}

      {!isLoading &&
        error === null &&
        conversations.length === 0 && (
          <p className="emptyHistory">
            Your conversations will appear here.
          </p>
        )}

      <div className="historyList">
        {conversations.map(
          (conversation) => {
            const isActive =
              conversation.id ===
              activeConversationId;

            return (
              <button
                type="button"
                key={conversation.id}
                className={
                  isActive
                    ? "historyItem activeHistoryItem"
                    : "historyItem"
                }
                onClick={() => {
                  onSelectConversation(
                    conversation.id,
                  );
                }}
              >
                <strong>
                  {conversation.title}
                </strong>

                <span>
                  {formatConversationDate(
                    conversation.updated_at,
                  )}
                  {" · "}
                  {conversation.message_count}{" "}
                  {conversation.message_count === 1
                    ? "message"
                    : "messages"}
                </span>
              </button>
            );
          },
        )}
      </div>
    </section>
  );
}