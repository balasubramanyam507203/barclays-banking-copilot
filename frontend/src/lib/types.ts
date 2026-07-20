export type ChatStatus =
  | "answered"
  | "abstained";

export type DevelopmentProfile =
  | "compliance_analyst"
  | "customer_support"
  | "security_investigator";

export type FeedbackRating =
  | "helpful"
  | "not_helpful";

export interface CurrentUser {
  subject: string;
  username: string;
  role: string;
  region: string;
  clearance_rank: number;
  groups: string[];
}

export interface AccessTokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: CurrentUser;
}

export interface SourceReference {
  label: string;
  document_id: string;
  title: string;
  version: string;
  chunk_id: string;
  source: string;
  citation: string;
}

export interface TokenUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
}

export interface GuardrailStatus {
  citation_validation_passed: boolean;
  post_generation_guardrails_passed: boolean;
  claims_checked: number;
  supported_claims: number;
}

export interface ChatResponse {
  request_id: string;

  conversation_id: string;
  user_message_id: string;
  assistant_message_id: string;

  status: ChatStatus;

  answer: string;
  abstained: boolean;
  model_called: boolean;

  citations_used: string[];
  sources: SourceReference[];

  evidence_count: number;

  guardrails: GuardrailStatus;
  usage: TokenUsage;
}

export interface FeedbackResponse {
  id: string;
  message_id: string;
  rating: FeedbackRating;
  comment?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationMessageRecord {
  id: string;
  role: "user" | "assistant";

  content: string;
  created_at: string;

  request_id?: string | null;
  status?: ChatStatus | null;

  abstained: boolean;
  model_called: boolean;

  citations_used: string[];
  sources: SourceReference[];

  evidence_count: number;

  guardrails?: GuardrailStatus | null;
  usage?: TokenUsage | null;

  feedback?: FeedbackResponse | null;
}

export interface ConversationDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;

  messages: ConversationMessageRecord[];
}

export interface SourceChunk {
  chunk_id: string;
  chunk_number: number;
  total_chunks: number;
  content: string;
}

export interface SourceDocument {
  document_id: string;
  title: string;
  version: string;
  source: string;
  chunks: SourceChunk[];
}

export interface UserChatMessage {
  id: string;
  role: "user";
  content: string;
}

export interface AssistantChatMessage {
  id: string;
  role: "assistant";
  content: string;

  response: ChatResponse;
  feedback: FeedbackResponse | null;
}

export type ChatMessage =
  | UserChatMessage
  | AssistantChatMessage;

export interface ApiValidationError {
  loc?: Array<string | number>;
  msg?: string;
  type?: string;
}

export interface ApiErrorBody {
  detail?: string | ApiValidationError[];
}