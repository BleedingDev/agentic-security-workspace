import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';
import { FlaskConical } from 'lucide-react';
import { withDocsBasePath } from './public-path';
import { appName, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span className="inline-flex items-center gap-2.5">
          <Image
            src={withDocsBasePath('/icon.svg')}
            width={28}
            height={28}
            alt=""
            aria-hidden="true"
            className="shrink-0"
          />
          <span>{appName}</span>
        </span>
      ),
    },
    links: [
      { text: 'Playground', url: 'https://wakarujs.com/playground/', icon: <FlaskConical /> },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
