html_layout = '''<!DOCTYPE html>
                    <html>
                    <link rel="stylesheet" media="screen" href = "\{\{ url_for('static', filename='bootstrap.min.css') }}">
                    <link rel="stylesheet" media="screen" href = "\{\{ url_for('static', filename='dist/css/styles.css') }}">
                    <meta name="viewport"  content = "width=device-width, initial-scale=1.0, user-scalable=yes">
                    <script src="https://kit.fontawesome.com/a10c0e47f2.js" crossorigin="anonymous"></script>
                        <head>
                            {%metas%}
                            <title>{%title%}</title>
                            {%favicon%}
                            {%css%}

                        </head>

                        <body>

                            <div class="navbar">
                                <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top py-3" id="mainNav">
                                    <a class="navbar-brand js-scroll-trigger" href="/">C.A.R.E</a>
                                    <span class="navbar-text navText">Breast Cancer Prediction Tool</span>

                                    <button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">
                                        <span class="navbar-toggler-icon"></span>
                                    </button>

                                    <div class="collapse navbar-collapse" id="navbarResponsive">
                                        <ul class="navbar-nav ml-auto">
                                            <li class="nav-item">
                                                <a class="nav-link js-scroll-trigger active" href="/index2.html"><i class="fas fa-calculator"></i> Calculator </a>
                                            </li>
                                            <li class="nav-item">
                                                <a class="nav-link js-scroll-trigger" href="/dashapp/"><i class="fas fa-chart-pie"></i> Dashboard</a>
                                            </li>
                                        </ul>
                                    </div>
                                </nav>
                            </div>


                            {%app_entry%}


                            <footer>
                                {%config%}
                                {%scripts%}
                                {%renderer%}
                            </footer>
                        </body>
                    </html>'''