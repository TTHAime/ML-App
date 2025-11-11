import type { PageLoad } from './$types';

export const load: PageLoad = ({ url }) => {
  const label = url.searchParams.get('label') ?? '';
  const prob = parseFloat(url.searchParams.get('prob') ?? '0');
  return { label, prob };
};
