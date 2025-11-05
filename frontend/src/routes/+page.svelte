<script lang="ts">
  import { goto } from '$app/navigation';
  import { PUBLIC_API_BASE } from '$env/static/public';

  let file: File | null = null;
  let previewUrl: string | null = null;
  let error = '';
  let uploading = false;
  let dragActive = false;

  let fileInput: HTMLInputElement | null = null;

  const MAX_SIZE_MB = 10;
  const ACCEPT = ['image/jpeg', 'image/png', 'image/webp'];

  function openFileDialog() {
    fileInput?.click();
  }

  function reset() {
    error = '';
    file = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  function onPick(e: Event) {
    const f = (e.currentTarget as HTMLInputElement).files?.[0] ?? null;
    validateAndSet(f);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragActive = false;
    const f = e.dataTransfer?.files?.[0] ?? null;
    validateAndSet(f);
  }

  function onDragOver(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragActive = true;
  }
  function onDragLeave() { dragActive = false; }

  function validateAndSet(f: File | null) {
    error = '';
    if (!f) return;
    if (!ACCEPT.includes(f.type)) { error = 'รองรับเฉพาะ .jpg .png .webp'; return; }
    const maxBytes = MAX_SIZE_MB * 1024 * 1024;
    if (f.size > maxBytes) { error = `ไฟล์ใหญ่เกิน ${MAX_SIZE_MB}MB`; return; }
    file = f;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f);
  }

  async function submit() {
    error = '';
    if (!file) { error = 'กรุณาเลือกไฟล์ภาพ'; return; }
    uploading = true;
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${PUBLIC_API_BASE}/predict`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const q = new URLSearchParams({
        label: data.label ?? '',
        prob: String(data.prob ?? ''),
        id: data.request_id ?? ''
      });
      goto(`/result?${q.toString()}`);
    } catch (e: any) {
      error = e?.message ?? 'อัปโหลดไม่สำเร็จ';
    } finally {
      uploading = false;
    }
  }
</script>

<div class="min-h-screen bg-linear-to-b from-slate-50 to-white dark:from-slate-950 dark:to-slate-900">
  <header class="mx-auto max-w-5xl px-6 pt-8">
    <div class="flex justify-center">
      <h1 class="text-center text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900 dark:text-slate-100">
        Cracked
      </h1>
    </div>
  </header>

  <main class="mx-auto max-w-xl px-6 py-10">
    <section class="rounded-3xl border border-slate-200/70 bg-white/70 shadow-sm backdrop-blur dark:border-slate-800 dark:bg-slate-900/60 p-6 sm:p-8">
      <div class="space-y-6">
        <div>
          <h2 class="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">อัปโหลดภาพกำแพง</h2>
          <p class="mt-1 text-sm text-slate-600 dark:text-slate-400">
            โมเดลจะทำนายว่า <span class="font-medium">crack</span> หรือ <span class="font-medium">no-crack</span>
            (รองรับ {ACCEPT.map((t) => t.split('/')[1]).join(', ')} สูงสุด {MAX_SIZE_MB}MB)
          </p>
        </div>

        <!-- Dropzone -->
        <div
          role="button"
          aria-label="พื้นที่วางไฟล์ภาพ; คลิกหรือกด Enter เพื่อเลือกไฟล์"
          tabindex="0"
          on:click|stopPropagation={openFileDialog}
          on:keydown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              openFileDialog();
            }
          }}
          on:drop|preventDefault|stopPropagation={onDrop}
          on:dragover|preventDefault|stopPropagation={onDragOver}
          on:dragleave={onDragLeave}
          class="relative rounded-2xl border-2 border-dashed p-6 transition
                 bg-slate-50/70 dark:bg-slate-800/40
                 border-slate-300/80 dark:border-slate-700
                 hover:border-slate-400 dark:hover:border-slate-600
                 focus:outline-none focus-visible:ring-4 focus-visible:ring-indigo-500/30
                 group cursor-pointer"
          class:border-indigo-500={dragActive || !!file}
        >
          <input
            id="file-input"
            bind:this={fileInput}
            type="file"
            accept={ACCEPT.join(',')}
            class="sr-only"
            on:change={onPick}
          />

          <div class="flex flex-col items-center justify-center gap-3 pointer-events-none">
            <div class="rounded-xl p-3 border border-slate-200/80 dark:border-slate-700/80 group-[.border-indigo-500]:border-indigo-300/70 group-[.border-indigo-500]:bg-indigo-50/40 dark:group-[.border-indigo-500]:bg-indigo-400/10">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="size-6 text-slate-500 dark:text-slate-400" fill="none" stroke="currentColor" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M7.5 7.5 12 3m0 0 4.5 4.5M12 3v12"/>
              </svg>
            </div>
            <p class="text-sm font-medium text-slate-800 dark:text-slate-200">
              ลากวางไฟล์ที่นี่ หรือ <span class="underline">คลิก/กด Enter เพื่อเลือกไฟล์</span>
            </p>
            <p class="text-xs text-slate-500 dark:text-slate-400">
              รองรับ {ACCEPT.map((t) => t.split('/')[1]).join(', ')} • สูงสุด {MAX_SIZE_MB}MB
            </p>
          </div>

          {#if previewUrl}
            <div class="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div class="flex items-center justify-between px-4 py-2 bg-slate-100/70 dark:bg-slate-800/60">
                <div class="text-sm truncate text-slate-700 dark:text-slate-300">
                  {file?.name} · {((file?.size ?? 0) / 1024 / 1024).toFixed(2)} MB
                </div>
                <button type="button" class="text-xs px-2 py-1 rounded-md border hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800 " on:click={reset} aria-label="ล้างไฟล์">
                  ล้างไฟล์
                </button>
              </div>
              <div class="aspect-video bg-white dark:bg-slate-900">
                <img src={previewUrl} alt="ตัวอย่างภาพ" class="h-full w-full object-contain" />
              </div>
            </div>
          {/if}
        </div>

        {#if error}
          <div class="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-900/20 dark:border-red-900/40 dark:text-red-300">
            {error}
          </div>
        {/if}

        <div class="flex items-center gap-3">
          <button class="px-4 py-2 rounded-xl bg-black text-white hover:opacity-90 disabled:opacity-50" on:click={submit} disabled={uploading || !file}>
            {uploading ? 'กำลังอัปโหลด…' : 'Predict'}
          </button>
          <button class="px-4 py-2 rounded-xl border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50  text-white" on:click={reset} disabled={uploading || !file}>
            ล้าง
          </button>
        </div>
      </div>
    </section>
  </main>
</div>
