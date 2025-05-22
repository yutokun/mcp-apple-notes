import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import * as lancedb from "@lancedb/lancedb";
import path from "node:path";
import os from "node:os";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const runAppleScript = async (script: string): Promise<string> => {
  try {
    const { stdout } = await execFileAsync('osascript', ['-e', script]);
    return stdout.trim();
  } catch (error) {
    console.error('AppleScript Error:', error);
    return '';
  }
};
import TurndownService from "turndown";
import {
  EmbeddingFunction,
  LanceSchema,
  register,
} from "@lancedb/lancedb/embedding";
import { type Float, Float32, Utf8 } from "apache-arrow";
import { pipeline } from "@huggingface/transformers";

const { turndown } = new TurndownService();
const db = await lancedb.connect(
  path.join(os.homedir(), ".mcp-apple-notes", "data")
);
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

@register("openai")
export class OnDeviceEmbeddingFunction extends EmbeddingFunction<string> {
  toJSON(): object {
    return {};
  }
  ndims() {
    return 384;
  }
  embeddingDataType(): Float {
    return new Float32();
  }
  async computeQueryEmbeddings(data: string) {
    const output = await extractor(data, { pooling: "mean" });
    return output.data as number[];
  }
  async computeSourceEmbeddings(data: string[]) {
    return await Promise.all(
      data.map(async (item) => {
        const output = await extractor(item, { pooling: "mean" });

        return output.data as number[];
      })
    );
  }
}

const func = new OnDeviceEmbeddingFunction();

const notesTableSchema = LanceSchema({
  title: func.sourceField(new Utf8()),
  content: func.sourceField(new Utf8()),
  creation_date: func.sourceField(new Utf8()),
  modification_date: func.sourceField(new Utf8()),
  vector: func.vectorField(),
});

const QueryNotesSchema = z.object({
  query: z.string(),
});

const GetNoteSchema = z.object({
  title: z.string(),
});

const server = new Server(
  {
    name: "my-apple-notes-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "list-notes",
        description: "Lists just the titles of all my Apple Notes",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "index-notes",
        description:
          "Index all my Apple Notes for Semantic Search. Please tell the user that the sync takes couple of seconds up to couple of minutes depending on how many notes you have.",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "get-note",
        description: "Get a note full content and details by title",
        inputSchema: {
          type: "object",
          properties: {
            title: z.string(),
          },
          required: ["title"],
        },
      },
      {
        name: "search-notes",
        description: "Search for notes by title or content",
        inputSchema: {
          type: "object",
          properties: {
            query: z.string(),
          },
          required: ["query"],
        },
      },
      {
        name: "create-note",
        description:
          "Create a new Apple Note with specified title and content. Must be in HTML format WITHOUT newlines",
        inputSchema: {
          type: "object",
          properties: {
            title: { type: "string" },
            content: { type: "string" },
          },
          required: ["title", "content"],
        },
      },
    ],
  };
});

const getNotes = async (): Promise<string[]> => {
  const script = `
    tell application "Notes"
      set noteList to {}
      set noteCount to count of notes
      repeat with i from 1 to noteCount
        set noteName to name of note i
        copy noteName to end of noteList
      end repeat
      return noteList
    end tell
  `;
  
  const result = await runAppleScript(script);
  // Convert the AppleScript list to a JavaScript array
  return result.split(',').map(item => item.trim());
};

const getNoteDetailsByTitle = async (title: string) => {
  const script = `
    function run(argv) {
      const Notes = Application('Notes');
      const title = argv[0];
      
      try {
        const notes = Notes.notes.whose({ name: title });
        if (notes.length === 0) {
          return JSON.stringify({ found: false, error: 'Note not found' });
        }
        
        const note = notes[0];
        return JSON.stringify({
          found: true,
          title: note.name(),
          content: note.body(),
          creation_date: note.creationDate().toISOString(),
          modification_date: note.modificationDate().toISOString()
        });
      } catch (e) {
        return JSON.stringify({ found: false, error: e.toString() });
      }
    }
  `;

  try {
    // osascript -l JavaScript を使用してJXAスクリプトを実行
    const result = await runJXAScript(script, [title]);
    const parsed = JSON.parse(result);
    
    if (!parsed.found) {
      throw new Error(parsed.error || `Note with title "${title}" not found`);
    }
    
    return {
      title: parsed.title,
      content: parsed.content,
      creation_date: parsed.creation_date,
      modification_date: parsed.modification_date
    };
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error(`Failed to parse JXA result: ${error.message}`);
    }
    throw error;
  }
};

// JXAスクリプトを実行するヘルパー関数
const runJXAScript = async (script: string, args: string[] = []): Promise<string> => {
  const fs = require('fs');
  const path = require('path');
  const os = require('os');
  const { execFile } = require('child_process');
  const { promisify } = require('util');
  const execFileAsync = promisify(execFile);
  
  // 一時ファイルにスクリプトを書き込む方法を使用
  const tempDir = os.tmpdir();
  const scriptPath = path.join(tempDir, `jxa_script_${Date.now()}.js`);
  
  try {
    fs.writeFileSync(scriptPath, script);
    
    const { stdout, stderr } = await execFileAsync('osascript', ['-l', 'JavaScript', scriptPath, ...args]);
    
    if (stderr) {
      throw new Error(`JXA execution error: ${stderr}`);
    }
    
    return stdout.trim();
  } finally {
    // 一時ファイルを削除
    try {
      fs.unlinkSync(scriptPath);
    } catch (e) {
      // 削除に失敗しても無視
    }
  }
};

export const indexNotes = async (notesTable: any) => {
  const start = performance.now();
  let report = "";
  
  console.error("Starting note indexing process...");
  
  const allNotes = (await getNotes()) || [];
  console.error(`Found ${allNotes.length} notes to process`);
  
  if (allNotes.length === 0) {
    return {
      chunks: 0,
      report: "No notes found",
      allNotes: 0,
      time: performance.now() - start,
    };
  }
  
  const batchSize = 10; // さらに小さなバッチサイズ
  const batches: string[][] = [];
  
  for (let i = 0; i < allNotes.length; i += batchSize) {
    batches.push(allNotes.slice(i, i + batchSize));
  }
  
  console.error(`Processing ${batches.length} batches of ${batchSize} notes each`);
  
  let processedCount = 0;
  const allNotesDetails: any[] = [];
  
  for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
    const batch = batches[batchIndex];
    console.error(`Processing batch ${batchIndex + 1}/${batches.length} (notes ${processedCount + 1}-${processedCount + batch.length})`);
    
    // 各ノートを順次処理（並列処理を避ける）
    for (const note of batch) {
      try {
        const noteDetails = await getNoteDetailsByTitle(note);
        if (noteDetails) {
          allNotesDetails.push(noteDetails);
        }
        processedCount++;
        
        if (processedCount % 50 === 0) {
          console.error(`Processed ${processedCount}/${allNotes.length} notes...`);
        }
      } catch (error: any) {
        console.error(`Error processing note "${note}": ${error.message}`);
        report += `Error processing note "${note}": ${error.message}\n`;
        processedCount++;
      }
    }
  }

  console.error(`Successfully processed ${allNotesDetails.length} notes, preparing chunks...`);

  const chunks = allNotesDetails
    .filter((n: any) => n && n.title)
    .map((note: any, index: number) => {
      try {
        const content = note.content || "";
        const markdownContent = content.includes('<') ? turndown(content) : content;
        
        return {
          id: index.toString(),
          title: note.title,
          content: markdownContent,
          creation_date: note.creation_date,
          modification_date: note.modification_date,
        };
      } catch (error) {
        console.error(`Processing error for note ${note.title}: ${error}`);
        return {
          id: index.toString(),
          title: note.title,
          content: note.content || "",
          creation_date: note.creation_date,
          modification_date: note.modification_date,
        };
      }
    });

  console.error(`Adding ${chunks.length} chunks to database...`);
  await notesTable.add(chunks);
  console.error("Database insertion completed");

  const totalTime = performance.now() - start;
  console.error(`Indexing completed in ${Math.round(totalTime)}ms`);

  return {
    chunks: chunks.length,
    report: "Indexing completed successfully",
    allNotes: allNotes.length,
    time: totalTime,
  };
};

export const createNotesTable = async (overrideName?: string) => {
  const start = performance.now();
  const notesTable = await db.createEmptyTable(
    overrideName || "notes",
    notesTableSchema,
    {
      mode: "create",
      existOk: true,
    }
  );

  const indices = await notesTable.listIndices();
  if (!indices.find((index) => index.name === "content_idx")) {
    await notesTable.createIndex("content", {
      config: lancedb.Index.fts(),
      replace: true,
    });
  }
  return { notesTable, time: performance.now() - start };
};

const createNote = async (title: string, content: string) => {
  // Escape special characters for AppleScript
  const escapedTitle = title.replace(/"/g, '\\"');
  const escapedContent = content
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\\\n')
    .replace(/\r/g, '');

  const script = `
    tell application "Notes"
      tell account "iCloud"
        make new note with properties {name:"${escapedTitle}", body:"${escapedContent}"}
      end tell
    end tell
    return true
  `;

  await runAppleScript(script);
  return true;
};

// Handle tool execution
server.setRequestHandler(CallToolRequestSchema, async (request, c) => {
  const { notesTable } = await createNotesTable();
  const { name, arguments: args } = request.params;

  try {
    if (name === "create-note") {
      const { title, content } = CreateNoteSchema.parse(args);
      await createNote(title, content);
      return createTextResponse(`Created note "${title}" successfully.`);
    } else if (name === "list-notes") {
      return createTextResponse(
        `There are ${await notesTable.countRows()} notes in your Apple Notes database.`
      );
    } else if (name == "get-note") {
      try {
        const { title } = GetNoteSchema.parse(args);
        const note = await getNoteDetailsByTitle(title);

        return createTextResponse(JSON.stringify(note, null, 2));
      } catch (error) {
        return createTextResponse(error.message);
      }
    } else if (name === "index-notes") {
      const { time, chunks, report, allNotes } = await indexNotes(notesTable);
      const message = `Indexed ${chunks} notes chunks from ${allNotes} total notes in ${Math.round(time)}ms. You can now search for them using the "search-notes" tool.`;
      return createTextResponse(message);
    } else if (name === "search-notes") {
      const { query } = QueryNotesSchema.parse(args);
      const combinedResults = await searchAndCombineResults(notesTable, query);
      return createTextResponse(JSON.stringify(combinedResults));
    } else {
      throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw new Error(
        `Invalid arguments: ${error.errors
          .map((e) => `${e.path.join(".")}: ${e.message}`)
          .join(", ")}`
      );
    }
    throw error;
  }
});

// Start the server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Local Machine MCP Server running on stdio");

const createTextResponse = (text: string) => ({
  content: [{ type: "text", text }],
});

/**
 * Search for notes by title or content using both vector and FTS search.
 * The results are combined using RRF
 */
export const searchAndCombineResults = async (
  notesTable: lancedb.Table,
  query: string,
  limit = 20
) => {
  const [vectorResults, ftsSearchResults] = await Promise.all([
    (async () => {
      const results = await notesTable
        .search(query, "vector")
        .limit(limit)
        .toArray();
      return results;
    })(),
    (async () => {
      const results = await notesTable
        .search(query, "fts", "content")
        .limit(limit)
        .toArray();
      return results;
    })(),
  ]);

  const k = 60;
  const scores = new Map<string, number>();

  const processResults = (results: any[], startRank: number) => {
    results.forEach((result, idx) => {
      const key = `${result.title}::${result.content}`;
      const score = 1 / (k + startRank + idx);
      scores.set(key, (scores.get(key) || 0) + score);
    });
  };

  processResults(vectorResults, 0);
  processResults(ftsSearchResults, 0);

  const results = Array.from(scores.entries())
    .sort(([, a], [, b]) => b - a)
    .slice(0, limit)
    .map(([key]) => {
      const [title, content] = key.split("::");
      return { title, content };
    });

  return results;
};

const CreateNoteSchema = z.object({
  title: z.string(),
  content: z.string(),
});
