export interface Citation {
  index: number;
  module: string;
  module_code?: string;
  week?: number | null;
  page_or_slide?: number | null;
  source_file: string;
  title?: string;
  text: string;
}

export interface AskResponse {
  answer: string;
  citations: Citation[];
  cached?: boolean;
}

export interface StreamToken {
  token?: string;
  answer?: string;
  citations?: Citation[];
  done?: boolean;
  error?: string;
}

export interface Recommendation {
  module: string;
  module_code?: string;
  week?: number | null;
  score: number;
  chunk_hits: number;
  sample_titles: string[];
}

export interface RecommendResponse {
  query: string;
  recommendations: Recommendation[];
}

export interface MCQOption {
  key: string;
  text: string;
}

export interface MCQQuestion {
  question: string;
  options: MCQOption[] | Record<string, string>;
  answer: string;
  explanation?: string;
}

export interface ShortAnswerQuestion {
  question: string;
  answer?: string;
  model_answer?: string;
  marks?: number;
}

export type QuizQuestion = MCQQuestion | ShortAnswerQuestion;

export interface QuizResponse {
  questions: QuizQuestion[];
  source_chunks: number;
}

export interface ExamQuestion {
  question: string;
  marks?: number;
  guidance?: string;
}

export interface ExamResponse {
  questions: ExamQuestion[];
}

export interface Workspace {
  workspace_id: string;
  name: string;
  token: string;
  doc_count: number;
  chunk_count: number;
}

export interface UploadResponse {
  document_id: number;
  job_id: number;
  status: string;
}

export interface DocumentStatus {
  id: number;
  filename: string;
  status: string;
  chunks: number;
  error?: string;
}

export interface WorkspaceStatus {
  workspace_id: string;
  documents: DocumentStatus[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  cached?: boolean;
}
