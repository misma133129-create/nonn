import { execFile } from 'node:child_process';
import crypto from 'node:crypto';
import dns from 'node:dns/promises';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { promisify } from 'node:util';
import { GoogleGenAI } from '@google/genai';
import type {
  QuranStoryAdminOverride,
  QuranStorySpec,
} from '../src/types/nonoland';

const execFileAsync = promisify(execFile);

export const STORY_AUDIO_PUBLIC_DIR = path.resolve(process.cwd(), 'public', 'assets', 'audio', 'stories');
const PUBLIC_ASSETS_ROOT = path.resolve(process.cwd(), 'public', 'assets');

export const STORY_NARRATION_TTS_MODEL = 'gemini-3.8-flash-lite-tts';
export const STORY_NARRATION_VOICE_NAME = 'Zephyr';

export const RELIGIOUS_REVIEW_FLAGS: Record<string, { reasonAr: string; reasonEn: string }> = {
  'ibrahim-s1': {
    reasonAr: 'يتضمن المشهد عبارة مبسّطة مقتبسة بالمعنى عن الآية 76 من سورة الأنعام؛ يُفضّل مراجعتها تحريرياً مع الإبقاء على الأصل بدون تغيير صامت.',
    reasonEn: 'Scene contains a child-simplified paraphrase of Al-Anam 6:76; flagged for editorial review rather than silent rewrite.',
  },
  'cave-s5': {
    reasonAr: 'يتضمن المشهد دعاء «ربنا آتنا من لدنك رحمة» ضمن السرد القصصي؛ يجب إبقاؤه بإلقاء قصصي تعليمي وليس تلاوة قرآنية.',
    reasonEn: 'Scene embeds the prayer wording from Al-Kahf 18:10 in narration; ensure conversational storytelling delivery rather than Quran recitation.',
  },
};

export interface StoryAudioAssetValidation {
  valid: boolean;
  url: string;
  kind: 'local' | 'remote' | 'invalid';
  status: 'valid' | 'missing' | 'invalid';
  mimeType: string | null;
  bytes: number;
  durationSeconds: number | null;
  reason?: string;
}

export interface StorySceneAudioManifestItem {
  storyId: string;
  storyTitleAr: string;
  storyTitleEn: string;
  sceneId: string;
  sceneOrder: number;
  sceneTitleAr: string;
  sceneTitleEn: string;
  narrationTextAr: string;
  narrationTextEn: string;
  expectedLocalPath: string;
  attachedUrl: string | null;
  assetStatus: 'valid' | 'missing' | 'invalid';
  bytes: number;
  durationSeconds: number | null;
  validationReason?: string;
  reviewFlag?: { reasonAr: string; reasonEn: string };
}

export interface StoryAudioManifestSummary {
  totalStories: number;
  totalScenes: number;
  totalRequiredAssets: number;
  validAudioCount: number;
  missingAudioCount: number;
  invalidAudioCount: number;
  ttsConfigured: boolean;
  ttsModel: string;
  voiceName: string;
  encodingTarget: string;
  items: StorySceneAudioManifestItem[];
}

function hasValidAudioMagicBytes(buf: Buffer): { ok: boolean; mimeType: string | null } {
  if (buf.length < 12) return { ok: false, mimeType: null };
  // MP3 with ID3 tag or MPEG frame sync
  if (buf.subarray(0, 3).toString('ascii') === 'ID3') {
    return { ok: true, mimeType: 'audio/mpeg' };
  }
  if (buf[0] === 0xff && (buf[1] & 0xe0) === 0xe0) {
    return { ok: true, mimeType: 'audio/mpeg' };
  }
  // RIFF WAVE
  if (
    buf.subarray(0, 4).toString('ascii') === 'RIFF' &&
    buf.subarray(8, 12).toString('ascii') === 'WAVE'
  ) {
    return { ok: true, mimeType: 'audio/wav' };
  }
  // Ogg
  if (buf.subarray(0, 4).toString('ascii') === 'OggS') {
    return { ok: true, mimeType: 'audio/ogg' };
  }
  // MP4 / M4A ftyp
  if (buf.subarray(4, 8).toString('ascii') === 'ftyp') {
    return { ok: true, mimeType: 'audio/mp4' };
  }
  return { ok: false, mimeType: null };
}

function isPrivateOrBlockedIp(ip: string): boolean {
  if (net.isIPv4(ip)) {
    const parts = ip.split('.').map(Number);
    if (parts[0] === 10 || parts[0] === 127 || parts[0] === 0) return true;
    if (parts[0] === 169 && parts[1] === 254) return true;
    if (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) return true;
    if (parts[0] === 192 && parts[1] === 168) return true;
    if (parts[0] === 100 && parts[1] >= 64 && parts[1] <= 127) return true;
    return false;
  }
  if (net.isIPv6(ip)) {
    const low = ip.toLowerCase();
    return (
      low === '::1' ||
      low === '::' ||
      low.startsWith('fc') ||
      low.startsWith('fd') ||
      low.startsWith('fe80') ||
      low.startsWith('::ffff:127.') ||
      low.startsWith('::ffff:10.') ||
      low.startsWith('::ffff:192.168.') ||
      low.startsWith('::ffff:169.254.')
    );
  }
  return true;
}

export function resolveSafePublicAssetFile(assetUrl: string): string | null {
  if (!assetUrl || !assetUrl.startsWith('/assets/') || assetUrl.startsWith('//')) return null;
  if (assetUrl.includes('\0') || assetUrl.includes('..')) return null;
  const relUnderAssets = assetUrl.replace(/^\/assets\//, '');
  const resolved = path.resolve(PUBLIC_ASSETS_ROOT, relUnderAssets);
  if (!resolved.startsWith(PUBLIC_ASSETS_ROOT + path.sep)) return null;
  return resolved;
}

export function getExpectedStoryAudioUrl(storyId: string, sceneId: string): string {
  return `/assets/audio/stories/${storyId}/${sceneId}.mp3`;
}

export function localStoryAudioExistsSync(assetUrl: string | null | undefined): boolean {
  if (!assetUrl) return false;
  if (!assetUrl.startsWith('/assets/')) return true; // remote URL verified separately on save
  const filePath = resolveSafePublicAssetFile(assetUrl);
  if (!filePath || !fs.existsSync(filePath)) return false;
  try {
    const stat = fs.statSync(filePath);
    if (!stat.isFile() || stat.size < 1024) return false;
    const fd = fs.openSync(filePath, 'r');
    try {
      const head = Buffer.alloc(16);
      const read = fs.readSync(fd, head, 0, 16, 0);
      if (read < 12) return false;
      return hasValidAudioMagicBytes(head).ok;
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return false;
  }
}

export async function probeLocalAudioDurationSeconds(filePath: string): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync(
      'ffprobe',
      [
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        filePath,
      ],
      { timeout: 6000 },
    );
    const parsed = Number.parseFloat(stdout.trim());
    if (Number.isFinite(parsed) && parsed > 0) {
      return Math.round(parsed * 10) / 10;
    }
  } catch {
    // fallback if WAV RIFF container
  }

  try {
    const buf = fs.readFileSync(filePath);
    if (buf.subarray(0, 4).toString('ascii') === 'RIFF' && buf.length > 44) {
      const byteRate = buf.readUInt32LE(28);
      if (byteRate > 0) {
        return Math.round(((buf.length - 44) / byteRate) * 10) / 10;
      }
    }
  } catch {
    // ignore
  }
  return null;
}

export async function validateStoryAudioAsset(rawUrl: string): Promise<StoryAudioAssetValidation> {
  const url = String(rawUrl || '').trim();
  if (!url) {
    return {
      valid: false,
      url: '',
      kind: 'invalid',
      status: 'missing',
      mimeType: null,
      bytes: 0,
      durationSeconds: null,
      reason: 'Audio URL is empty.',
    };
  }

  if (url.startsWith('/assets/')) {
    const filePath = resolveSafePublicAssetFile(url);
    if (!filePath) {
      return {
        valid: false,
        url,
        kind: 'local',
        status: 'invalid',
        mimeType: null,
        bytes: 0,
        durationSeconds: null,
        reason: 'Invalid local /assets/ audio path.',
      };
    }
    if (!fs.existsSync(filePath)) {
      return {
        valid: false,
        url,
        kind: 'local',
        status: 'missing',
        mimeType: null,
        bytes: 0,
        durationSeconds: null,
        reason: 'Audio file does not exist on disk.',
      };
    }
    const stat = fs.statSync(filePath);
    if (!stat.isFile() || stat.size < 1024 || stat.size > 15 * 1024 * 1024) {
      return {
        valid: false,
        url,
        kind: 'local',
        status: 'invalid',
        mimeType: null,
        bytes: stat.size,
        durationSeconds: null,
        reason: 'Local audio file is empty or outside valid size limits (1 KB – 15 MB).',
      };
    }
    const fd = fs.openSync(filePath, 'r');
    const head = Buffer.alloc(16);
    try {
      fs.readSync(fd, head, 0, 16, 0);
    } finally {
      fs.closeSync(fd);
    }
    const magic = hasValidAudioMagicBytes(head);
    if (!magic.ok) {
      return {
        valid: false,
        url,
        kind: 'local',
        status: 'invalid',
        mimeType: null,
        bytes: stat.size,
        durationSeconds: null,
        reason: 'File header is not a recognized audio format (MP3/WAV/OGG/M4A).',
      };
    }
    const duration = await probeLocalAudioDurationSeconds(filePath);
    if (duration !== null && (duration < 0.4 || duration > 300)) {
      return {
        valid: false,
        url,
        kind: 'local',
        status: 'invalid',
        mimeType: magic.mimeType,
        bytes: stat.size,
        durationSeconds: duration,
        reason: `Measured audio duration (${duration}s) is outside allowed story scene range (0.4s–300s).`,
      };
    }
    return {
      valid: true,
      url,
      kind: 'local',
      status: 'valid',
      mimeType: magic.mimeType,
      bytes: stat.size,
      durationSeconds: duration,
    };
  }

  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return {
      valid: false,
      url,
      kind: 'invalid',
      status: 'invalid',
      mimeType: null,
      bytes: 0,
      durationSeconds: null,
      reason: 'Malformed URL.',
    };
  }

  if (parsed.protocol !== 'https:' || parsed.username || parsed.password) {
    return {
      valid: false,
      url,
      kind: 'invalid',
      status: 'invalid',
      mimeType: null,
      bytes: 0,
      durationSeconds: null,
      reason: 'Only clean HTTPS URLs or /assets/... paths are permitted.',
    };
  }

  const host = parsed.hostname.toLowerCase();
  if (host === 'localhost' || host.endsWith('.local')) {
    return {
      valid: false,
      url,
      kind: 'remote',
      status: 'invalid',
      mimeType: null,
      bytes: 0,
      durationSeconds: null,
      reason: 'Private or localhost URLs are blocked.',
    };
  }

  try {
    const records = await dns.lookup(host, { all: true });
    if (!records.length || records.some((r) => isPrivateOrBlockedIp(r.address))) {
      return {
        valid: false,
        url,
        kind: 'remote',
        status: 'invalid',
        mimeType: null,
        bytes: 0,
        durationSeconds: null,
        reason: 'Remote host resolves to a restricted IP address.',
      };
    }
  } catch {
    return {
      valid: false,
      url,
      kind: 'remote',
      status: 'invalid',
      mimeType: null,
      bytes: 0,
      durationSeconds: null,
      reason: 'Remote hostname could not be resolved.',
    };
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const res = await fetch(url, {
      method: 'GET',
      headers: { Range: 'bytes=0-16383' },
      redirect: 'follow',
      signal: controller.signal,
    });
    if (!res.ok) {
      return {
        valid: false,
        url,
        kind: 'remote',
        status: 'missing',
        mimeType: res.headers.get('content-type'),
        bytes: 0,
        durationSeconds: null,
        reason: `Remote audio URL returned HTTP ${res.status}.`,
      };
    }
    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    const buf = Buffer.from(await res.arrayBuffer());
    const magic = hasValidAudioMagicBytes(buf);
    if (!magic.ok && !contentType.startsWith('audio/')) {
      return {
        valid: false,
        url,
        kind: 'remote',
        status: 'invalid',
        mimeType: contentType || null,
        bytes: buf.length,
        durationSeconds: null,
        reason: `Remote URL did not return valid audio data (content-type=${contentType || 'unknown'}).`,
      };
    }
    const contentLen = Number(res.headers.get('content-length') || buf.length);
    return {
      valid: true,
      url,
      kind: 'remote',
      status: 'valid',
      mimeType: magic.mimeType || contentType || 'audio/mpeg',
      bytes: Number.isFinite(contentLen) && contentLen > 0 ? contentLen : buf.length,
      durationSeconds: null,
    };
  } catch (err) {
    return {
      valid: false,
      url,
      kind: 'remote',
      status: 'invalid',
      mimeType: null,
      bytes: 0,
      durationSeconds: null,
      reason: err instanceof Error ? err.message : 'Remote audio probe failed.',
    };
  } finally {
    clearTimeout(timeout);
  }
}

export async function buildStoryAudioManifest(
  baseStories: QuranStorySpec[],
  overrides: Record<string, QuranStoryAdminOverride>,
): Promise<StoryAudioManifestSummary> {
  const items: StorySceneAudioManifestItem[] = [];
  let validAudioCount = 0;
  let missingAudioCount = 0;
  let invalidAudioCount = 0;

  for (const story of baseStories) {
    const ov = overrides[story.id];
    const orderList = ov?.sceneOrder && ov.sceneOrder.length > 0 ? ov.sceneOrder : story.scenes.map((s) => s.id);
    const sceneMap = new Map(story.scenes.map((s) => [s.id, s]));
    const orderedScenes = orderList
      .map((id) => sceneMap.get(id))
      .filter((sc): sc is QuranStorySpec['scenes'][number] => Boolean(sc));

    for (const sc of orderedScenes) {
      const scOv = ov?.sceneOverrides?.[sc.id];
      const expectedLocalPath = getExpectedStoryAudioUrl(story.id, sc.id);
      const explicitUrl = scOv?.narrationAudioUrl;
      let candidateUrl: string | null = null;
      if (explicitUrl === null) {
        candidateUrl = null;
      } else if (typeof explicitUrl === 'string' && explicitUrl.trim()) {
        candidateUrl = explicitUrl.trim();
      } else if (sc.narrationAudioUrl) {
        candidateUrl = sc.narrationAudioUrl;
      } else if (localStoryAudioExistsSync(expectedLocalPath)) {
        candidateUrl = expectedLocalPath;
      }

      let assetStatus: 'valid' | 'missing' | 'invalid' = 'missing';
      let bytes = 0;
      let durationSeconds: number | null = null;
      let validationReason: string | undefined;

      if (candidateUrl) {
        if (candidateUrl.startsWith('/assets/')) {
          const v = await validateStoryAudioAsset(candidateUrl);
          assetStatus = v.status;
          bytes = v.bytes;
          durationSeconds = v.durationSeconds;
          validationReason = v.reason;
        } else {
          // For pre-validated remote URLs saved via Admin PUT, treat format check fast in manifest
          assetStatus = 'valid';
          durationSeconds = scOv?.durationSeconds ?? sc.durationSeconds;
        }
      } else {
        assetStatus = 'missing';
        validationReason = 'No validated Arabic narration recording attached yet.';
      }

      if (assetStatus === 'valid') validAudioCount++;
      else if (assetStatus === 'invalid') invalidAudioCount++;
      else missingAudioCount++;

      items.push({
        storyId: story.id,
        storyTitleAr: ov?.title?.ar || story.title.ar,
        storyTitleEn: ov?.title?.en || story.title.en,
        sceneId: sc.id,
        sceneOrder: sc.order,
        sceneTitleAr: scOv?.title?.ar || sc.title.ar,
        sceneTitleEn: scOv?.title?.en || sc.title.en,
        narrationTextAr: scOv?.narration?.mid?.ar || sc.narration.mid.ar,
        narrationTextEn: scOv?.narration?.mid?.en || sc.narration.mid.en,
        expectedLocalPath,
        attachedUrl: candidateUrl,
        assetStatus,
        bytes,
        durationSeconds,
        validationReason,
        reviewFlag: RELIGIOUS_REVIEW_FLAGS[sc.id],
      });
    }
  }

  return {
    totalStories: baseStories.length,
    totalScenes: items.length,
    totalRequiredAssets: items.length,
    validAudioCount,
    missingAudioCount,
    invalidAudioCount,
    ttsConfigured: Boolean(process.env.GEMINI_API_KEY),
    ttsModel: STORY_NARRATION_TTS_MODEL,
    voiceName: STORY_NARRATION_VOICE_NAME,
    encodingTarget: 'MP3 64kbps mono 24kHz',
    items,
  };
}

function ensureWavBufferFromGeminiPart(base64Audio: string, mimeType?: string): Buffer {
  const raw = Buffer.from(base64Audio, 'base64');
  if (raw.subarray(0, 4).toString('ascii') === 'RIFF') {
    return raw;
  }
  const mt = (mimeType || '').toLowerCase();
  let sampleRate = 24000;
  const rateMatch = mt.match(/rate=(\d+)/);
  if (rateMatch) {
    const parsed = Number(rateMatch[1]);
    if (Number.isFinite(parsed) && parsed >= 8000 && parsed <= 48000) sampleRate = parsed;
  }
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const header = Buffer.alloc(44);
  header.write('RIFF', 0, 'ascii');
  header.writeUInt32LE(36 + raw.length, 4);
  header.write('WAVE', 8, 'ascii');
  header.write('fmt ', 12, 'ascii');
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36, 'ascii');
  header.writeUInt32LE(raw.length, 40);
  return Buffer.concat([header, raw]);
}

export async function transcodeWavBufferToMp3File(
  wavBuffer: Buffer,
  destMp3Path: string,
): Promise<{ bytes: number; durationSeconds: number | null }> {
  const destDir = path.dirname(destMp3Path);
  fs.mkdirSync(destDir, { recursive: true });
  const tmpWav = path.join(destDir, `.tmp-${crypto.randomUUID()}.wav`);
  const tmpMp3 = path.join(destDir, `.tmp-${crypto.randomUUID()}.mp3`);
  try {
    fs.writeFileSync(tmpWav, wavBuffer);
    await execFileAsync(
      'ffmpeg',
      [
        '-y',
        '-i',
        tmpWav,
        '-ar',
        '24000',
        '-ac',
        '1',
        '-codec:a',
        'libmp3lame',
        '-b:a',
        '64k',
        tmpMp3,
      ],
      { timeout: 20000 },
    );
    const stat = fs.statSync(tmpMp3);
    if (stat.size < 1024) {
      throw new Error('Encoded MP3 audio is unexpectedly small.');
    }
    const durationSeconds = await probeLocalAudioDurationSeconds(tmpMp3);
    fs.renameSync(tmpMp3, destMp3Path);
    return { bytes: stat.size, durationSeconds };
  } finally {
    if (fs.existsSync(tmpWav)) fs.unlinkSync(tmpWav);
    if (fs.existsSync(tmpMp3)) fs.unlinkSync(tmpMp3);
  }
}

export async function generateSceneNarrationAudio(opts: {
  storyId: string;
  sceneId: string;
  narrationTextAr: string;
  voiceName?: string;
}): Promise<{
  url: string;
  bytes: number;
  durationSeconds: number | null;
  mimeType: string;
  model: string;
  voiceName: string;
}> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    const err = new Error('Server narration audio generation requires GEMINI_API_KEY.') as Error & {
      code?: string;
      statusCode?: number;
    };
    err.code = 'TTS_PROVIDER_NOT_CONFIGURED';
    err.statusCode = 503;
    throw err;
  }

  const cleanText = String(opts.narrationTextAr || '').trim();
  if (!cleanText || cleanText.length > 900) {
    const err = new Error('Scene Arabic narration text must be 1–900 characters.') as Error & {
      code?: string;
      statusCode?: number;
    };
    err.code = 'INVALID_NARRATION_TEXT';
    err.statusCode = 400;
    throw err;
  }

  const allowedVoices = new Set(['Puck', 'Charon', 'Kore', 'Fenrir', 'Zephyr']);
  const chosenVoice = opts.voiceName && allowedVoices.has(opts.voiceName) ? opts.voiceName : STORY_NARRATION_VOICE_NAME;

  const ai = new GoogleGenAI({
    apiKey,
    httpOptions: { headers: { 'User-Agent': 'aistudio-build' } },
  });

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await ai.models.generateContent({
      model: STORY_NARRATION_TTS_MODEL,
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: cleanText,
              speechMetadata: {
                style:
                  'Warm, gentle, loving Arabic storyteller with clear Egyptian-friendly pronunciation for toddlers ages 1 to 4. Calm joyful conversational narration, clear articulation, never chanting or Quran recitation.',
              },
            },
          ],
        },
      ],
      config: {
        abortSignal: controller.signal,
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: chosenVoice },
          },
        },
      },
    });

    const inlinePart = response.candidates?.[0]?.content?.parts?.[0]?.inlineData;
    if (!inlinePart?.data) {
      const err = new Error('Audio synthesis returned an empty payload.') as Error & {
        code?: string;
        statusCode?: number;
      };
      err.code = 'TTS_EMPTY_AUDIO';
      err.statusCode = 502;
      throw err;
    }

    const wavBuffer = ensureWavBufferFromGeminiPart(inlinePart.data, inlinePart.mimeType);
    const expectedUrl = getExpectedStoryAudioUrl(opts.storyId, opts.sceneId);
    const destPath = resolveSafePublicAssetFile(expectedUrl);
    if (!destPath) {
      throw new Error('Invalid destination audio path.');
    }

    const encoded = await transcodeWavBufferToMp3File(wavBuffer, destPath);
    const check = await validateStoryAudioAsset(expectedUrl);
    if (!check.valid) {
      if (fs.existsSync(destPath)) fs.unlinkSync(destPath);
      throw new Error(check.reason || 'Generated MP3 failed post-write validation.');
    }

    return {
      url: expectedUrl,
      bytes: encoded.bytes,
      durationSeconds: check.durationSeconds ?? encoded.durationSeconds,
      mimeType: 'audio/mpeg',
      model: STORY_NARRATION_TTS_MODEL,
      voiceName: chosenVoice,
    };
  } finally {
    clearTimeout(timeout);
  }
}
