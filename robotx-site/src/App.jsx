import React, { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Heart,
  Handshake,
  Info,
  Mail,
  Ship,
  Plus,
  Trash2,
  ExternalLink,
} from "lucide-react";

/**
 * Single-file React site (Tailwind + shadcn/ui).
 * - Home (RobotX description)
 * - Become a Sponsor (GoFundMe donate section)
 * - Materials Needed (editable list)
 * - About Us (editable team members)
 * - Contact Us
 *
 * Notes:
 * - Replace GOFUNDME_URL with your campaign link.
 * - This is front-end only; forms are demo-only (no backend).
 */

const GOFUNDME_URL = "https://gofund.me/316a2c928"; // TODO: replace with your campaign

const NAV = [
  { id: "home", label: "RobotX" },
  { id: "sponsor", label: "Become a Sponsor" },
  { id: "materials", label: "Needed Materials" },
  { id: "about", label: "About Us" },
  { id: "contact", label: "Contact" },
];

const fadeUp = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35 } },
};

function Section({ id, title, icon: Icon, children }) {
  return (
    <section id={id} className="scroll-mt-24">
      <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="h-10 w-10 rounded-2xl bg-black/5 dark:bg-white/10 flex items-center justify-center">
            <Icon className="h-5 w-5" />
          </div>
          <h2 className="text-2xl md:text-3xl font-semibold tracking-tight">{title}</h2>
        </div>
        {children}
      </motion.div>
    </section>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-2xl border bg-background p-4">
      <div className="text-sm text-muted-foreground">{label}</div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
  );
}

function SponsorTier({ name, amount, perks }) {
  return (
    <Card className="rounded-2xl">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-3">
          <span>{name}</span>
          <Badge variant="secondary" className="rounded-xl">{amount}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2 text-sm">
          {perks.map((p, i) => (
            <li key={i} className="flex gap-2">
              <span className="mt-1 h-1.5 w-1.5 rounded-full bg-foreground/40" />
              <span className="text-muted-foreground">{p}</span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function MaterialsEditor() {
  const items = [
    { name: "Tickets to Singapore", qty: "Assorted", priority: "High" },
    { name: "Marine-grade hardware (stainless fasteners)", qty: "Assorted", priority: "High" },
    { name: "Batteries / power distribution components", qty: "TBD", priority: "High" },
    { name: "Sensors (GPS, IMU, cameras)", qty: "TBD", priority: "Medium" },
    { name: "Comms (telemetry radios, antennas)", qty: "TBD", priority: "Medium" },
    { name: "Safety equipment (kill switch, floatation)", qty: "TBD", priority: "High" },
  ];

  const priorityPill = (p) => {
    const cls =
      p === "High"
        ? "bg-foreground text-background"
        : p === "Medium"
        ? "bg-black/10 dark:bg-white/10"
        : "bg-black/5 dark:bg-white/5";
    return <span className={`px-2 py-1 rounded-xl text-xs ${cls}`}>{p}</span>;
  };

  return (
    <div className="space-y-3">
      {items.map((it, idx) => (
        <div
          key={idx}
          className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 rounded-2xl border p-4"
        >
          <div className="min-w-0">
            <div className="font-medium truncate">{it.name}</div>
            <div className="text-sm text-muted-foreground mt-1">Qty: {it.qty || "—"}</div>
          </div>
          {priorityPill(it.priority)}
        </div>
      ))}
    </div>
  );
}

function TeamEditor() {
  const team = [
    {
      name: "Samantha Garcia",
      role: "UAV Team Lead",
      bio: "",
    },
    {
      name: "Xavier Vicent Navarro",
      role: "USV/UUV Team Lead",
      bio: "",
    },
  ];

  return (
    <div className="grid md:grid-cols-2 gap-4">
      {team.map((m, idx) => (
        <Card key={idx} className="rounded-2xl">
          <CardHeader>
            <CardTitle>
              <div className="truncate">{m.name}</div>
              <div className="text-sm text-muted-foreground font-normal mt-1">{m.role || "—"}</div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground leading-relaxed">{m.bio || ""}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function ContactForm() {
  const [form, setForm] = useState({ name: "", email: "", message: "" });
  const [sent, setSent] = useState(false);

  const submit = (e) => {
    e.preventDefault();
    setSent(true);
    setTimeout(() => setSent(false), 2500);
    setForm({ name: "", email: "", message: "" });
  };

  return (
      <Card className="rounded-2xl">
        <CardHeader>
          <CardTitle>Contact details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <div>
            <div className="text-muted-foreground">General email</div>
            <div className="font-medium">birdsim20@gmail.com</div>
          </div>
          <div>
            <div className="text-muted-foreground">Sponsorship</div>
            <div className="font-medium">scruz10@fau.edu</div>
          </div>
        </CardContent>
      </Card>
  );
}

export default function RobotXWebsite() {
  const [active, setActive] = useState("home");

  const heroBullets = useMemo(
    () => [
      "Autonomous maritime systems: perception, navigation, safety, mission planning.",
      "Real-world engineering: hardware, software, systems integration, testing.",
      "Public impact: STEM outreach, sponsor visibility, and community engagement.",
    ],
    []
  );

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    setActive(id);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <div className="sticky top-0 z-50 backdrop-blur supports-[backdrop-filter]:bg-background/70 bg-background/90 border-b">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
          <button
            onClick={() => scrollTo("home")}
            className="flex items-center gap-2 min-w-0"
            aria-label="Go to top"
          >
            <div className="h-9 w-9 rounded-2xl bg-black/5 dark:bg-white/10 flex items-center justify-center">
              <Ship className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="font-semibold leading-none truncate">RobotX Team</div>
              <div className="text-xs text-muted-foreground leading-none mt-1 truncate">UAV • USV • UUV</div>
            </div>
          </button>

          <nav className="hidden md:flex items-center gap-1">
            {NAV.map((n) => (
              <Button
                key={n.id}
                variant={active === n.id ? "default" : "ghost"}
                className="rounded-2xl"
                onClick={() => scrollTo(n.id)}
              >
                {n.label}
              </Button>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <Button
              className="rounded-2xl"
              onClick={() => scrollTo("sponsor")}
            >
              <Heart className="h-4 w-4 mr-2" /> Sponsor
            </Button>
          </div>
        </div>

        {/* Mobile nav */}
        <div className="md:hidden max-w-6xl mx-auto px-4 pb-3">
          <div className="flex gap-2 overflow-x-auto no-scrollbar">
            {NAV.map((n) => (
              <Button
                key={n.id}
                variant={active === n.id ? "default" : "secondary"}
                className="rounded-2xl whitespace-nowrap"
                onClick={() => scrollTo(n.id)}
              >
                {n.label}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Hero */}
      <div id="home" className="scroll-mt-24">
        <div className="max-w-6xl mx-auto px-4 py-10 md:py-14">
          <div className="grid gap-8 lg:grid-cols-2 items-center">
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
              <Badge variant="secondary" className="rounded-xl">RobotX Competition</Badge>
              <h1 className="text-3xl md:text-5xl font-semibold tracking-tight mt-3">
                Building autonomous systems for RobotX — and beyond.
              </h1>
              <p className="text-muted-foreground mt-4 leading-relaxed">
                RobotX is an international competition that challenges teams to design and operate autonomous maritime robots
                in complex, real-world scenarios. Our team builds platforms and software spanning perception, navigation,
                safety, and mission execution.
              </p>

              <ul className="mt-5 space-y-2">
                {heroBullets.map((b, i) => (
                  <li key={i} className="flex gap-3 text-sm">
                    <span className="mt-2 h-1.5 w-1.5 rounded-full bg-foreground/50" />
                    <span className="text-muted-foreground">{b}</span>
                  </li>
                ))}
              </ul>

              <div className="mt-6 flex flex-wrap gap-3">
                <Button className="rounded-2xl" onClick={() => scrollTo("sponsor")}
                >
                  <Handshake className="h-4 w-4 mr-2" /> Become a Sponsor
                </Button>
                <Button variant="secondary" className="rounded-2xl" onClick={() => scrollTo("materials")}
                >
                  <Plus className="h-4 w-4 mr-2" /> View Needed Materials
                </Button>
              </div>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay: 0.05 }}>
              <Card className="rounded-2xl">
                <CardHeader>
                  <CardTitle>At a glance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid sm:grid-cols-2 gap-3">
                    <Stat label="Tracks" value="UAV • USV • UUV" />
                    <Stat label="Focus" value="Autonomy + Integration" />
                    <Stat label="Sponsors" value="Industry + Community" />
                    <Stat label="Impact" value="STEM + Innovation" />
                  </div>
                  <div className="mt-4 rounded-2xl border p-4">
                    <div className="text-sm font-medium">What we’re building</div>
                    <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
                      A competition-ready autonomous maritime platform, with reliable perception, state estimation,
                      mission planning, and robust safety systems — tested in realistic environments.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-4 pb-16 space-y-14">
        <Section id="sponsor" title="Become a Sponsor" icon={Handshake}>
          <div className="grid gap-6 lg:grid-cols-3">
            <Card className="rounded-2xl lg:col-span-2">
              <CardHeader>
                <CardTitle>Support our season</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Sponsorship helps us cover competition fees, travel, safety equipment, sensors, compute, fabrication,
                  and on-water testing. Sponsors receive visibility on our website, social media, and vehicle branding
                  (based on tier).
                </p>

                <div className="rounded-2xl border p-4">
                  <div className="flex items-center justify-between gap-3 flex-wrap">
                    <div>
                      <div className="font-medium">Donate via GoFundMe</div>
                      <div className="text-sm text-muted-foreground mt-1">Every contribution directly supports the team.</div>
                    </div>
                    <Button asChild className="rounded-2xl">
                      <a href={GOFUNDME_URL} target="_blank" rel="noreferrer">
                        <Heart className="h-4 w-4 mr-2" /> Donate <ExternalLink className="h-4 w-4 ml-2" />
                      </a>
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground mt-3">
                    Replace the link at the top of the file (GOFUNDME_URL) with your actual campaign.
                  </div>
                </div>

                <div className="grid md:grid-cols-3 gap-4">
                  <SponsorTier
                    name="Bronze"
                    amount="$250+"
                    perks={["Logo on website", "Social media thank-you", "Sponsor updates"]}
                  />
                  <SponsorTier
                    name="Silver"
                    amount="$1,000+"
                    perks={["Bronze perks", "Logo on team shirts", "Vehicle logo placement (small)"]}
                  />
                  <SponsorTier
                    name="Gold"
                    amount="$5,000+"
                    perks={["Silver perks", "Vehicle logo placement (large)", "Demo day invite"]}
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="rounded-2xl">
              <CardHeader>
                <CardTitle>Other ways to help</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="rounded-2xl border p-4">
                  <div className="font-medium">In-kind donations</div>
                  <div className="text-muted-foreground mt-1">Hardware, fabrication support, test access, and services.</div>
                </div>
                <div className="rounded-2xl border p-4">
                  <div className="font-medium">Mentorship</div>
                  <div className="text-muted-foreground mt-1">Autonomy, marine engineering, safety, and systems reviews.</div>
                </div>
                
                <Button variant="secondary" className="rounded-2xl w-full" onClick={() => {
                  const el = document.getElementById("contact");
                  el?.scrollIntoView({ behavior: "smooth", block: "start" });
                  setActive("contact");
                }}>
                  <Mail className="h-4 w-4 mr-2" /> Contact us
                </Button>
              </CardContent>
            </Card>
          </div>
        </Section>

        <Section id="materials" title="Needed Materials" icon={Plus}>
          <p className="text-sm text-muted-foreground leading-relaxed mb-6">
            Below are examples of materials and equipment that help support the team and our RobotX platform.
          </p>
          <MaterialsEditor />
        </Section>

        <Section id="about" title="About Us" icon={Info}>
          <div className="space-y-6">
            <Card className="rounded-2xl">
              <CardHeader>
                <CardTitle>Who we are</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  We’re a multidisciplinary team building autonomous systems for the RobotX competition. Our work spans
                  mechanical design, electrical integration, autonomy software, perception, and field testing. Below are some of our team members working on the RobotX platform.
                </p>
              </CardContent>
            </Card>
            <TeamEditor />
          </div>
        </Section>

        <Section id="contact" title="Contact Us" icon={Mail}>
          <p className="text-sm text-muted-foreground leading-relaxed mb-6">
            Want to sponsor, mentor, collaborate, or learn more? Send us a message and we’ll respond quickly.
          </p>
          <ContactForm />
        </Section>

        <footer className="pt-10 border-t">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 py-6">
            <div className="text-sm text-muted-foreground">
              © {new Date().getFullYear()} RobotX Team. All rights reserved.
            </div>
            <div className="flex flex-wrap gap-2">
              <Button variant="ghost" className="rounded-2xl" onClick={() => scrollTo("home")}>Top</Button>
              <Button variant="ghost" className="rounded-2xl" onClick={() => scrollTo("sponsor")}>Sponsor</Button>
              <Button variant="ghost" className="rounded-2xl" onClick={() => scrollTo("materials")}>Materials</Button>
              <Button variant="ghost" className="rounded-2xl" onClick={() => scrollTo("contact")}>Contact</Button>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
