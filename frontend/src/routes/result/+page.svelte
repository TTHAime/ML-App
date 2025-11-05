<script lang="ts">
  // รับ data จาก load (ไม่ต้อง import { page } อีก)
  export let data: { label: string; prob: string; id: string };
  const { label, prob, id } = data;
</script>

<div class="min-h-screen flex items-center justify-center p-6">
  <div class="w-full max-w-xl space-y-4">
    <h1 class="text-2xl font-bold">Surface Crack: Result</h1>
    <div class="rounded border p-4">
      <p class="text-sm opacity-70">Request ID: {id}</p>
      <p class="text-lg">Prediction: <b>{label}</b></p>
      <p class="text-sm">
        Confidence: {(parseFloat(prob) * 100).toFixed(2)}%
      </p>
    </div>
    <a href="/" class="underline">&larr; Upload again</a>
  </div>
</div>
