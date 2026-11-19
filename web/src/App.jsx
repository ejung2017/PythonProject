import './App.css'

// Add a project by adding an entry here.
const projects = [
  {
    title: 'WWDC2025 Siri News Sentiment Analysis',
    href: 'https://github.com/ejung2017/PythonProject/tree/main/project1',
    blurb: `Scraped 98 news articles on Apple Intelligence Siri and scored sentiment to
            measure public expectation. 35% optimistic on personalization, 65%
            disappointed by repeated delays.`,
    tags: ['NLP', 'SpaCy', 'Web Scraping', 'WordCloud'],
    media: { type: 'image', src: '/projects/siri-wordcloud.png',
             alt: 'Word cloud of the most frequent terms across 98 Siri news articles' },
  },
  {
    title: 'Stock Prediction Web App',
    href: 'https://github.com/ejung2017/PythonProject/tree/main/project2',
    blurb: `Predicts up/down price trends from technical indicators (MA, RSI, MACD,
            Bollinger Bands), keeping only models above 80% accuracy with under a 5%
            overfitting gap, then simulates 3-month profit/loss from their signals.`,
    tags: ['Scikit-learn', 'Streamlit', 'TA-Lib', 'Statsmodels'],
    demo: { label: 'Live app', href: 'https://eunahjung.app/stock/' },
    media: { type: 'video', src: '/projects/stock-demo.mp4' },
  },
  {
    title: 'Amazon Product Review Analysis',
    href: 'https://github.com/ejung2017/PythonProject/tree/main/project3',
    blurb: `Crawled and compared reviews for two competing products to pick the better
            buy. Product A scored 78% positive against B's 52%, winning on durability,
            design, and convenience.`,
    tags: ['Selenium', 'SpaCy', 'WordCloud'],
    media: { type: 'pair', src: ['/projects/reviews-nicwell.png', '/projects/reviews-coslus.png'],
             alt: ['Sentiment breakdown for Product A (Nicwell)',
                   'Sentiment breakdown for Product B (COSLUS)'] },
  },
  {
    title: 'LinkedIn Job Scraping',
    href: 'https://github.com/ejung2017/PythonProject/tree/main/project4',
    blurb: `Scrapes Data Scientist postings in Hong Kong — company, title, date, and
            application URL — into a CSV for tracking the local market.`,
    tags: ['Selenium', 'Pandas'],
  },
]

const skills = [
  ['Languages', 'Python, SQL, JavaScript'],
  ['ML & Stats', 'Scikit-learn, Statsmodels, TA-Lib, time-series forecasting'],
  ['NLP', 'SpaCy, sentiment analysis, WordCloud'],
  ['Data', 'Pandas, NumPy, Seaborn, Matplotlib'],
  ['Scraping', 'Selenium, BeautifulSoup, Requests'],
  ['Tools', 'React, Streamlit, Jupyter, Git, GitHub Actions, nginx'],
]

function Media({ type, src, alt }) {
  if (type === 'video') {
    // 4MB file — preload only metadata so the card costs nothing until played.
    return (
      <video className="shot" src={src} controls preload="metadata" playsInline />
    )
  }
  if (type === 'pair') {
    return (
      <div className="shot-pair">
        {src.map((s, i) => <img className="shot" key={s} src={s} alt={alt[i]} loading="lazy" />)}
      </div>
    )
  }
  return <img className="shot" src={src} alt={alt} loading="lazy" />
}

function ProjectCard({ title, href, blurb, tags, demo, media }) {
  return (
    <article className="card">
      <h3><a href={href}>{title}</a></h3>
      <p>{blurb}</p>
      <ul className="tags">
        {tags.map((t) => <li key={t}>{t}</li>)}
      </ul>
      {demo && <a className="demo" href={demo.href}>{demo.label} →</a>}
      {media && <Media {...media} />}
    </article>
  )
}

export default function App() {
  return (
    <main className="wrap">
      <header>
        <h1>Eunah Jung</h1>
        <p className="lede">
          Python, machine learning, and NLP projects — sentiment analysis,
          time-series forecasting, and web scraping applied to real market questions.
        </p>
        <p className="links">
          <a href="https://github.com/ejung2017/PythonProject">GitHub</a>
          <a href="mailto:jungeunah98@gmail.com">Email</a>
        </p>
      </header>

      <h2>About</h2>
      <p>
        I build end-to-end data projects: scraping the raw source, cleaning it,
        modeling it, and shipping something someone can actually click. My focus is
        machine learning and NLP — sentiment, forecasting, and turning messy public
        text into a decision.
      </p>

      <h2>Skills</h2>
      <dl className="skills">
        {skills.map(([label, value]) => (
          <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
        ))}
      </dl>

      <h2>Projects</h2>
      {projects.map((p) => <ProjectCard key={p.title} {...p} />)}

      <footer>Built with React. Deployed from main by GitHub Actions.</footer>
    </main>
  )
}
