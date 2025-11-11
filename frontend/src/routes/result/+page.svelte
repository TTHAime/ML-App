<script lang="ts">
  export let data: { label: string; prob: number };
  const { label, prob } = data;
  const pct = (prob * 100).toFixed(2);
</script>

<!-- เข้มสุด: พื้นหลัง #0a1024 ไล่ไปเกือบดำ, ตัวอักษรขาว -->
<div class="min-h-screen text-white
            bg-gradient-to-b from-[#0a1024] via-[#050a1a] to-[#030713]
            flex items-center justify-center p-6 selection:bg-white/10">
  <div class="w-full max-w-xl space-y-7">

    <h1 class="text-3xl font-semibold tracking-tight drop-shadow-[0_2px_8px_rgba(255,255,255,0.15)]">
      Surface Crack: Result
    </h1>

    <!-- การ์ดเข้มจัด โปร่งน้อย ขอบขาวชัด -->
    <div class="rounded-2xl border-2 border-white/60 bg-white/[0.03] backdrop-blur-sm
                p-6 shadow-[0_10px_40px_rgba(0,0,0,0.7)] ring-1 ring-white/10">
      <p class="text-xl">
        Prediction:
        <span class="ml-2 font-semibold px-2.5 py-1 rounded-lg
                     bg-white/10 border border-white/50">
          {label}
        </span>
      </p>

      <!-- Confidence bar (เข้ม/คอนทราสต์สูง) -->
      <div class="mt-5">
        <div class="flex items-center justify-between text-sm text-white/90 mb-1.5">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div class="h-3 w-full rounded-full bg-white/10 overflow-hidden border border-white/40">
          <div
            class="h-full bg-white"
            style={`width:${pct}%`}
            role="progressbar"
            aria-valuenow={parseFloat(pct)}
            aria-valuemin="0"
            aria-valuemax="100"
          />
        </div>
      </div>
    </div>

    <a href="/"
       class="inline-flex items-center gap-2 text-white hover:text-white
              underline underline-offset-4 decoration-white/80 hover:decoration-white transition">
      &larr; Upload again
    </a>
  </div>
</div>
