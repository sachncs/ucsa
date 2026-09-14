import { Nav } from './components/Nav';
import { Footer } from './components/Footer';
import { Hero } from './sections/Hero';
import { Concept } from './sections/Concept';
import { Architecture } from './sections/Architecture';
import { Banks } from './sections/Banks';
import { ReasoningLoop } from './sections/ReasoningLoop';
import { Proof } from './sections/Proof';
import { Quickstart } from './sections/Quickstart';
import { Docs } from './sections/Docs';
import { Research } from './sections/Research';
import { Limitations } from './sections/Limitations';
import { Contributing } from './sections/Contributing';
import { Closing } from './sections/Closing';

export default function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Concept />
        <Architecture />
        <Banks />
        <ReasoningLoop />
        <Proof />
        <Quickstart />
        <Docs />
        <Research />
        <Limitations />
        <Contributing />
        <Closing />
      </main>
      <Footer />
    </>
  );
}