// The JS client of Chroma (@chroma-core/chroma-cloud-qwen and chroma-cloud-splade) with CHROMA_EMBED_URL pointing at
// ChromaDotNet.Embeddings.Server, against the vectors that Chroma Cloud recorded on 25 October 2026.
// It fails beyond the tolerances of the tests of ChromaDotNet.Embeddings: 1 - cosine 1e-5 for Qwen, values 0.06 apart for SPLADE.
//   cd build/server-clients && npm ci && CHROMA_EMBED_URL=http://127.0.0.1:8080 CHROMA_API_KEY=... node js_client_check.mjs <cloud-v2.json>
import { readFileSync } from "node:fs";
import { ChromaCloudQwenEmbeddingFunction, ChromaCloudQwenEmbeddingModel } from "@chroma-core/chroma-cloud-qwen";
import { ChromaCloudSpladeEmbeddingFunction } from "@chroma-core/chroma-cloud-splade";

const cloud = JSON.parse(readFileSync(process.argv[2], "utf8"));
const cosine = (a, b) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / Math.sqrt(x * y); };
const worst = (mine, theirs) => Math.max(...mine.map((v, i) => 1 - cosine(v, theirs[i])));

const qwen = new ChromaCloudQwenEmbeddingFunction({ model: ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B, task: "nl_to_code" });
const check = (ok, what) => { if (!ok) { console.error(`FAILED: ${what}`); process.exitCode = 1; } };
const documents = await qwen.generate(cloud["qwen-documents"].request.texts);
const documentsWorst = worst(documents, cloud["qwen-documents"].vectors);
console.log(`qwen documents: ${documents.length}, 1 - cosine at most ${documentsWorst.toExponential(2)}`);
check(documentsWorst <= 1e-5, "qwen documents");
const queries = await qwen.generateForQueries(cloud["qwen-queries"].request.texts);
const queriesWorst = worst(queries, cloud["qwen-queries"].vectors);
console.log(`qwen queries with the instruction: 1 - cosine at most ${queriesWorst.toExponential(2)}`);
check(queriesWorst <= 1e-5, "qwen queries");

for (const includeTokens of [false, true]) {
  const splade = new ChromaCloudSpladeEmbeddingFunction({ includeTokens });
  let max = 0;
  for (const name of ["splade-documents", "splade-queries"]) {
    const vectors = await splade.generate(cloud[name].request.texts);
    vectors.forEach((v, i) => {
      const remote = new Map(cloud[name].vectors[i].indices.map((index, j) => [index, cloud[name].vectors[i].values[j]]));
      v.indices.forEach((index, j) => { max = Math.max(max, Math.abs(v.values[j] - (remote.get(index) ?? 0))); });
    });
    if (includeTokens && name === "splade-queries") console.log(`splade labels of the first query: ${vectors[0].labels?.slice(0, 4)}`);
  }
  console.log(`splade, includeTokens ${includeTokens}: values up to ${max.toExponential(2)} from those of Chroma Cloud`);
  check(max <= 0.06, `splade, includeTokens ${includeTokens}`);
}
try {
  await qwen.generate(["word ".repeat(9000)]);
  check(false, "a text of 9000 words is refused");
} catch (e) {
  console.log(`a text of 9000 words: ${e.message.slice(0, 160)}`);
}
