import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { Composer, MarkdownView } from "./main";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("renders lyrics as a labeled verse with literal text and stanza breaks", () => {
  const lyrics = "*A light beside the door*\nOne more night\n\n<script>Stay</script>";
  const { container } = render(
    <MarkdownView markdownBlocks={[
      { kind: "code_block", language: "lyrics", text: lyrics },
      { kind: "code_block", language: "python", text: "play()" }
    ]} />
  );

  const verse = screen.getByRole("figure", { name: "Sung lyrics" });
  expect(verse.querySelector(".lyrics-text")?.textContent).toBe(lyrics);
  expect(verse.querySelector("code, script, em")).toBeNull();
  expect(container.querySelector("pre code")?.textContent).toBe("play()");
});

function renderComposer() {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ id: "job-1", type: "chat_turn", status: "queued", result: null, error: null })
  });
  vi.stubGlobal("fetch", fetchMock);
  render(
    <QueryClientProvider client={new QueryClient()}>
      <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
    </QueryClientProvider>
  );
  return {
    textarea: screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement,
    fetchMock
  };
}

it("fences selected composer lines as lyrics and submits the original verse", async () => {
  const { textarea, fetchMock } = renderComposer();
  fireEvent.change(textarea, { target: { value: "*I sing.*\nA light\n\nOne more night\n*I pause.*" } });
  textarea.setSelectionRange(12, 32);
  await userEvent.click(screen.getByRole("button", { name: "Format as lyrics" }));

  const expected = "*I sing.*\n```lyrics\nA light\n\nOne more night\n```\n*I pause.*";
  expect(textarea).toHaveValue(expected);
  await userEvent.click(screen.getByTitle("Send"));
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
  const call = fetchMock.mock.calls.find(([path]) => path === "/api/chat");
  expect(JSON.parse(String(call?.[1].body))).toMatchObject({ body: expected, save_id: "save-1" });
});

it("inserts an empty lyrics block using the focused composer shortcut", async () => {
  const { textarea } = renderComposer();
  textarea.focus();
  await userEvent.keyboard("{Alt>}l{/Alt}");

  expect(textarea).toHaveValue("```lyrics\n\n```");
  await waitFor(() => expect(textarea.selectionStart).toBe(10));
  expect(textarea.selectionEnd).toBe(10);
});

it("toggles an existing lyrics block from a caret inside the verse", async () => {
  const { textarea } = renderComposer();
  fireEvent.change(textarea, { target: { value: "*I sing.*\n```lyrics\nA light\n\nOne more night\n```\n*I pause.*" } });
  textarea.setSelectionRange(22, 22);
  await userEvent.click(screen.getByRole("button", { name: "Format as lyrics" }));

  expect(textarea).toHaveValue("*I sing.*\nA light\n\nOne more night\n*I pause.*");
});

it("clears lyrics fences while preserving literal lyric punctuation", async () => {
  const { textarea } = renderComposer();
  fireEvent.change(textarea, { target: { value: "*I sing.*\n```lyrics\n*Stay*\n> with me\n```\n```python\nplay()\n```" } });
  textarea.setSelectionRange(0, textarea.value.length);
  await userEvent.click(screen.getByRole("button", { name: "Clear roleplay formatting" }));

  expect(textarea).toHaveValue("I sing.\n*Stay*\n> with me\n```python\nplay()\n```");
});

it("toggles a lyrics selection that includes the newline after its closing fence", async () => {
  const { textarea } = renderComposer();
  const block = "```lyrics\n*Stay*\n> with me\n```";
  fireEvent.change(textarea, { target: { value: `${block}\n*I pause.*` } });
  textarea.setSelectionRange(0, block.length + 1);
  await userEvent.click(screen.getByRole("button", { name: "Format as lyrics" }));

  expect(textarea).toHaveValue("*Stay*\n> with me\n*I pause.*");
});

it.each([
  [0, 25, "I sing.\n*Stay*\n> with me\n*I pause.*"],
  [22, 51, "*I sing.*\n*Stay*\n> with me\nI pause."]
])("clears a selection crossing a lyrics boundary without damaging the verse (%i, %i)", async (start, end, expected) => {
  const { textarea } = renderComposer();
  fireEvent.change(textarea, { target: { value: "*I sing.*\n```lyrics\n*Stay*\n> with me\n```\n*I pause.*" } });
  textarea.setSelectionRange(start, end);
  await userEvent.click(screen.getByRole("button", { name: "Clear roleplay formatting" }));

  expect(textarea).toHaveValue(expected);
});
