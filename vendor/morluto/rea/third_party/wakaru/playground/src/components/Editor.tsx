import MonacoEditor, { type OnMount } from "@monaco-editor/react";
import { useCallback, useEffect, useRef } from "react";
import type { editor as MonacoEditorNS } from "monaco-editor";
import { EditorPaneHeader } from "./EditorPaneHeader";

export interface EditorDecoration {
  line: number;
  startCol: number;
  endCol: number;
  className: string;
  wholeLine?: boolean;
}

interface EditorProps {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  decorations?: EditorDecoration[];
  onHoverLine?: (line: number | null) => void;
  onEditorReady?: (editor: MonacoEditorNS.IStandaloneCodeEditor) => void;
}

export function Editor({
  value,
  onChange,
  label = "Input",
  decorations,
  onHoverLine,
  onEditorReady,
}: EditorProps) {
  const editorRef = useRef<MonacoEditorNS.IStandaloneCodeEditor | null>(null);
  const decorationIds = useRef<string[]>([]);
  const hoverRef = useRef(onHoverLine);
  hoverRef.current = onHoverLine;

  const handleMount: OnMount = useCallback((editor) => {
    editorRef.current = editor;
    onEditorReady?.(editor);
    editor.onMouseMove((e) => {
      const line = e.target.position?.lineNumber ?? null;
      hoverRef.current?.(line !== null ? line - 1 : null);
    });
    editor.onMouseLeave(() => hoverRef.current?.(null));
  }, []);

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    if (!decorations || decorations.length === 0) {
      decorationIds.current = editor.deltaDecorations(decorationIds.current, []);
      return;
    }
    const monacoDecorations: MonacoEditorNS.IModelDeltaDecoration[] = decorations.map((d) => ({
      range: {
        startLineNumber: d.line + 1,
        startColumn: d.startCol + 1,
        endLineNumber: d.line + 1,
        endColumn: d.endCol + 1,
      },
      options: d.wholeLine
        ? { className: d.className, isWholeLine: true }
        : { inlineClassName: d.className },
    }));
    decorationIds.current = editor.deltaDecorations(decorationIds.current, monacoDecorations);
  }, [decorations]);

  return (
    <div className="editor-pane">
      <EditorPaneHeader className="editor-pane-label">
        {label}
      </EditorPaneHeader>
      <MonacoEditor
        language="javascript"
        theme="vs-dark"
        value={value}
        onChange={(v) => onChange(v ?? "")}
        onMount={handleMount}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          scrollBeyondLastLine: false,
          wordWrap: "on",
          automaticLayout: true,
          padding: { top: 12 },
        }}
      />
    </div>
  );
}
