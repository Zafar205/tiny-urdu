import "./globals.css";

export const metadata = { title: "Tiny Urdu", description: "Roman Urdu text-to-speech in the browser (tiny-voca)" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
