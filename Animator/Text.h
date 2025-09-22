#pragma once

#include <GL/glew.h>
#include <freetype/freetype.h>

#include <string>
#include <vector>
#include <ranges>

#include "Geometry.h"

class Text {
public:
	
	QuadManager::QuadReference mQuadReference;

	GLuint mTexture = 0;
	FT_Int mTotalPixelsWidth = 0, mTotalPixelsHeight = 0, mMaxAscent = 0, mMaxDescent = 0;
	std::vector<std::uint32_t> mTextPixelsBuffer;

	void create( const std::string& text, const FT_Face& face) {

		mTotalPixelsWidth = 0;
		mMaxAscent = 0;
		mMaxDescent = 0;
		
		for (auto c: text ){

			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;

			mTotalPixelsWidth += g->advance.x >> 6; // advance in pixels
			mMaxAscent = std::max(mMaxAscent, g->bitmap_top);
			mMaxDescent = std::max(mMaxDescent, FT_Int(g->bitmap.rows - g->bitmap_top));
		}
		mTotalPixelsHeight = mMaxAscent + mMaxDescent;

		std::uint32_t greyColor = 0xFFFFFFFF;//opaque with white background with grey forground/text
		std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(&greyColor);

		//add spaces between glyphs and padding of line and bottom of line
		mTotalPixelsWidth += text.size()*2 +2 ;
		mTotalPixelsHeight += 2;
		mTextPixelsBuffer.clear();
		mTextPixelsBuffer.resize(mTotalPixelsWidth * mTotalPixelsHeight, greyColor);

		int penX = 0;
		for (std::size_t i : std::views::iota(0ULL, text.size())) {

			char c = text[i];

			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;
			FT_Bitmap& bmp = g->bitmap;

			int xOffset = penX + g->bitmap_left;
			int yOffset = mMaxAscent - g->bitmap_top;

			for (int row = 0; row < bmp.rows; ++row) {
				for (int col = 0; col < bmp.width; ++col) {
					int x = 1 + xOffset + col + i*2; //+i/1 == space between glyphs
					int y = 1 + yOffset + row;

					float greyScale = 1.0f - bmp.buffer[row * bmp.pitch + col] / 255.0f;
					std::uint8_t* savedColor = reinterpret_cast<std::uint8_t*>(&mTextPixelsBuffer[y * mTotalPixelsWidth + x]);

					savedColor[0] = bytes[0] * greyScale; // R
					savedColor[1] = bytes[1] * greyScale;   // G
					savedColor[2] = bytes[2] * greyScale; // B
					savedColor[3] = bytes[3]; // A
				}
			}

			penX += g->advance.x >> 6;
		}
	
		if (mTexture)
			glDeleteTextures(1, &mTexture);

		glGenTextures(1, &mTexture);

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, mTexture);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mTotalPixelsWidth, mTotalPixelsHeight, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, mTextPixelsBuffer.data());

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void destroy(){
		if (mTexture) {
			glDeleteTextures(1, &mTexture);
			mTexture = 0;
		}
		mTextPixelsBuffer.clear();
		mTextPixelsBuffer.shrink_to_fit();
	}
	
	float getAspectRatio() {
		return mTotalPixelsWidth / float(mTotalPixelsHeight);
	}
};

class TextManager {

public:

	QuadManager* mQuadManager;

	FT_Library mFreeType;
	FT_Face mFontFace;

	std::vector<Text> mLabels;
	using LabelReference = std::size_t;

	using CaptionValue = std::pair<Text, Text>;
	std::vector<CaptionValue> mCaptionValues;
	using CaptionValueReference = std::size_t;

	float mScale = 0.01f;

	float mInsertY = -1.0f; //start at bottom
	bool mVisible = true;

	void create(QuadManager* quadManager, const std::string& fontName, FT_UInt fontSize, float scale = 0.01f) {

		mQuadManager = quadManager;
		mScale = scale;

		FT_Init_FreeType(&mFreeType);

		FT_New_Face(mFreeType, fontName.c_str(), 0, &mFontFace);
		FT_Set_Pixel_Sizes(mFontFace, 0, fontSize);
	}

	void destroy() {

		FT_Done_Face(mFontFace);       // Destroys the font face
		FT_Done_FreeType(mFreeType);     // Shuts down the FreeType library

		for (auto& label : mLabels)
			label.destroy();

		for (auto& [ caption,value] : mCaptionValues) {

			caption.destroy();
			value.destroy();
		}
	}

	void raiseYInsert(float distance) {
		mInsertY += distance;
	}

	LabelReference addLabel(const std::string& text) {
		Text t;
		t.create(text, mFontFace);
		t.mQuadReference = mQuadManager->add(-1.0, mInsertY, t.getAspectRatio(), mScale);

		mLabels.push_back(std::move(t));

		raiseYInsert(2.0f * mScale);
		return mLabels.size() - 1;
	}

	CaptionValueReference addLabeledValue(const std::string& text, const std::string& valueText) {

		float insertX = -1;

		Text caption;
		caption.create(text, mFontFace);
		caption.mQuadReference = mQuadManager->add(insertX, mInsertY, caption.getAspectRatio(), mScale);

		insertX += 2.0 * caption.getAspectRatio() * mScale;

		Text value;
		value.create(valueText, mFontFace);
		value.mQuadReference = mQuadManager->add(insertX, mInsertY, value.getAspectRatio(), mScale);

		mCaptionValues.push_back({ std::move(caption), std::move(value) });

		raiseYInsert(2.0f * mScale);

		return mCaptionValues.size() - 1;
	}
	
	void updateCaptionValue(CaptionValueReference captionValueRef, const std::string& valueText) {
		auto& value = mCaptionValues[captionValueRef].second;
		value.create(valueText, mFontFace);
	}
	void updateLabel(LabelReference labelRef, const std::string& labelText) {
		auto& label = mLabels[labelRef];
		label.create(labelText, mFontFace);
	}

	void render() {

		if (!mVisible) return;

		auto& qm = *mQuadManager;
		 
		//glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		for (auto& text : mLabels) {

			glBindTexture(GL_TEXTURE_2D, text.mTexture);
			qm.render(text.mQuadReference);
		}

		for (auto& [text, value] : mCaptionValues) {

			glBindTexture(GL_TEXTURE_2D, text.mTexture);
			qm.render(text.mQuadReference);

			glBindTexture(GL_TEXTURE_2D, value.mTexture);
			qm.render(value.mQuadReference);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
	} 
	void show() {
		mVisible = true;
	}
	void hide() {
		mVisible = false;
	}
	void toggleVisibility() {
		mVisible = !mVisible;
	}
};