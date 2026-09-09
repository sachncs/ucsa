source "https://rubygems.org"

# Jekyll and dependencies pinned to GitHub Pages' supported versions
# (https://pages.github.com/versions/). Run ``bundle install`` then
# ``bundle exec jekyll serve`` to preview the site locally.
gem "github-pages", group: :jekyll_plugins

# Plugins listed in _config.yml under ``plugins:``.
group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-remote-theme"
end

# Windows and JRuby do not include zoneinfo files
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]
